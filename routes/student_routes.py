from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError
from database import get_db
from pydantic import BaseModel
from async_lru import alru_cache
from typing import Dict, Any
import logging

# Setup logger
logger = logging.getLogger("student_routes")
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/student", tags=["Student"])

# Pydantic model for quiz submission
class QuizSubmission(BaseModel):
    student_name: str
    student_usn: str
    test_id: int
    total_marks: int
    max_marks: int
    time_taken: int

# ----------------------------
# Caching for test questions
# ----------------------------
@alru_cache(maxsize=128)
async def get_test_data_cached(test_id: int, db: AsyncSession) -> Dict[str, Any]:
    """
    Fetch test schema, questions, and top rows with caching per test_id.
    """
    # Fetch test info
    test_res = await db.execute(
        text("SELECT table_name, schema_sql FROM test WHERE id = :id"), {"id": test_id}
    )
    test = test_res.fetchone()
    if not test:
        raise HTTPException(status_code=404, detail="Test not found")

    table_name, schema_sql = test

    # Fetch questions
    q_res = await db.execute(
        text(
            "SELECT id, question_text, difficulty, expected_sql "
            "FROM question WHERE test_id = :tid"
        ),
        {"tid": test_id},
    )
    questions = [
        {"id": q[0], "question_text": q[1], "difficulty": q[2], "expected_sql": q[3]}
        for q in q_res.fetchall()
    ]

    # Fetch top rows (preview)
    try:
        data_res = await db.execute(text(f"SELECT * FROM {table_name} LIMIT 5"))
        top_rows = [dict(r._mapping) for r in data_res.fetchall()]
    except Exception as e:
        logger.warning(f"Failed to fetch top rows for {table_name}: {e}")
        top_rows = []

    return {
        "schema_sql": schema_sql,
        "questions": questions,
        "top_rows": top_rows,
    }

# ----------------------------
# Endpoints
# ----------------------------
@router.get("/teachers/{teacher_id}/tests")
async def get_tests_by_teacher(teacher_id: int, db: AsyncSession = Depends(get_db)):
    """
    Get all tests created by a specific teacher.
    """
    try:
        res = await db.execute(
            text("SELECT id, name, description FROM test WHERE teacher_id = :tid"), {"tid": teacher_id}
        )
        tests = [{"id": t[0], "name": t[1], "description": t[2]} for t in res.fetchall()]
        return {"tests": tests}
    except Exception as e:
        logger.error(f"Error fetching tests for teacher {teacher_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch tests")

@router.get("/tests/{test_id}/questions")
async def get_test_questions(test_id: int, db: AsyncSession = Depends(get_db)):
    """
    Get test schema, preview data, and questions.
    Cached per test_id for efficiency.
    """
    try:
        test_data = await get_test_data_cached(test_id, db)
        return test_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching test questions for test_id {test_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch test data")

@router.post("/tests/{test_id}/submissions")
async def submit_quiz(sub: QuizSubmission, db: AsyncSession = Depends(get_db)):
    """
    Store student quiz submission safely.
    Prevents duplicate submissions per student per test.
    """
    try:
        student_usn = sub.student_usn.upper()
        # Check duplicate submission
        check = await db.execute(
            text("SELECT id FROM quiz_submission WHERE student_usn = :usn AND test_id = :tid"),
            {"usn": student_usn, "tid": sub.test_id}
        )
        if check.fetchone():
            return {"message": "Already submitted", "status": "duplicate"}

        # Insert submission
        res = await db.execute(
            text("""
                INSERT INTO quiz_submission
                (student_name, student_usn, test_id, total_marks, max_marks, time_taken, submitted_at)
                VALUES (:name, :usn, :tid, :marks, :max, :time, NOW())
                RETURNING id
            """),
            {
                "name": sub.student_name,
                "usn": student_usn,
                "tid": sub.test_id,
                "marks": sub.total_marks,
                "max": sub.max_marks,
                "time": sub.time_taken
            }
        )
        submission_id = res.scalar_one()
        await db.commit()
        logger.info(f"Submission stored for student {student_usn}, test {sub.test_id}")
        return {"message": "Submission stored", "submission_id": submission_id, "status": "created"}

    except IntegrityError:
        await db.rollback()
        logger.warning(f"Duplicate submission attempt for student {student_usn}, test {sub.test_id}")
        return {"message": "Duplicate submission", "status": "duplicate"}
    except Exception as e:
        await db.rollback()
        logger.error(f"Error storing submission for student {student_usn}, test {sub.test_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to store submission")
