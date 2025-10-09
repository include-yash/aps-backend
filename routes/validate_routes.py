from fastapi import APIRouter, Depends, Body, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from database import get_db
from pydantic import BaseModel
import logging
import asyncio
from typing import List, Tuple
from functools import lru_cache


router = APIRouter(prefix="/validate", tags=["Validation"])

# Logger setup
logger = logging.getLogger("validate_logger")
logging.basicConfig(level=logging.INFO)

# Simple in-memory cache for expected results per question_id
EXPECTED_CACHE_SIZE = 500  # adjust as needed
@lru_cache(maxsize=EXPECTED_CACHE_SIZE)
def cached_expected_sql(question_id: int, expected_rows: Tuple[Tuple]):
    return expected_rows


class ValidateInput(BaseModel):
    test_id: int
    question_id: int
    user_sql: str


async def execute_select_safe(db: AsyncSession, sql: str, timeout: int = 5):
    """
    Safely execute a SELECT query with timeout.
    Raises HTTPException for invalid SQL or execution errors.
    """
    try:
        # Enforce query timeout
        async def run_query():
            result = await db.execute(text(sql))
            if not result.returns_rows:
                raise HTTPException(status_code=400, detail="SQL must be a SELECT query")
            return result.fetchall()

        return await asyncio.wait_for(run_query(), timeout=timeout)

    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Query execution timed out")
    except Exception as e:
        msg = str(e)
        if "column" in msg.lower() and "does not exist" in msg.lower():
            raise HTTPException(status_code=400, detail="Invalid column in query")
        elif "syntax" in msg.lower() or "parse" in msg.lower():
            raise HTTPException(status_code=400, detail="SQL syntax error")
        else:
            # Mask sensitive internal details
            logger.error(f"SQL execution error: {msg}")
            raise HTTPException(status_code=400, detail="SQL execution failed")


@router.post("/")
async def validate_answer(payload: ValidateInput, db: AsyncSession = Depends(get_db)):
    # Input validation
    if not payload.user_sql.strip():
        raise HTTPException(status_code=400, detail="User SQL cannot be empty")
    if len(payload.user_sql) > 5000:
        raise HTTPException(status_code=400, detail="User SQL too long")

    # Fetch expected SQL from DB
    try:
        res = await db.execute(
            text("SELECT expected_sql FROM question WHERE id = :qid AND test_id = :tid"),
            {"qid": payload.question_id, "tid": payload.test_id}
        )
        row = res.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Question not found")
        expected_sql = row[0]
    except Exception as e:
        logger.error(f"Error fetching expected SQL: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch question")

    # Check cache
    cache_key = (payload.question_id, expected_sql)
    try:
        expected_rows = cached_expected_sql(payload.question_id, tuple())
        if not expected_rows:
            expected_rows = await execute_select_safe(db, expected_sql)
            cached_expected_sql.cache_clear()  # update cache
            cached_expected_sql(payload.question_id, tuple(expected_rows))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing expected SQL: {e}")
        raise HTTPException(status_code=500, detail="Failed to execute expected SQL")

    # Execute user SQL safely
    user_rows = await execute_select_safe(db, payload.user_sql)

    # Normalize results for comparison
    def normalize(rows: List[Tuple]):
        return sorted([tuple(r) for r in rows])

    expected_norm = normalize(expected_rows)
    user_norm = normalize(user_rows)
    is_correct = expected_norm == user_norm

    logger.info(f"Validation result test_id={payload.test_id}, question_id={payload.question_id}: {is_correct}")

    return {
        "is_correct": is_correct,
        "expected_output": expected_norm,
        "user_output": user_norm
    }
