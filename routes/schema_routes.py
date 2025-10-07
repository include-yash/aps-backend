import os
import json
from fastapi import APIRouter, Body, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from database import get_db

# Imports for Gemini API and environment variables
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv
from sqlalchemy.exc import IntegrityError


# Load environment variables from .env file
load_dotenv()

# Configure the Gemini API with your key
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")
genai.configure(api_key=GEMINI_API_KEY)


router = APIRouter()


# Add this Pydantic model for the submission data
class QuizSubmission(BaseModel):
    student_name: str
    student_usn: str
    test_id: int
    total_marks: int
    max_marks: int
    time_taken: int  # in seconds
class SchemaUpload(BaseModel):
    test_name: str
    description: str | None = None
    schema: str  # full CREATE TABLE statement
    num_questions: int = 10 # Optional: specify number of questions

# Helper function to generate data using Gemini
async def generate_insert_statements(schema: str, table_name: str, num_rows: int = 15) -> str:
    """
    Uses the Gemini API to generate realistic SQL INSERT statements for a given table schema.
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = f"""
    Based on the following SQL table schema, please generate {num_rows} realistic and diverse `INSERT` statements to populate the table.
    The output should be ONLY a single, valid SQL script containing the INSERT statements.
    Each statement must end with a semicolon.
    Do not include any other text, explanations, or markdown formatting like ```sql.

    Schema:
    ```sql
    {schema}
    ```
    """
    try:
        response = await model.generate_content_async(prompt)
        # Clean up the response to ensure it's valid SQL
        generated_sql = response.text.strip()
        if not generated_sql.upper().startswith("INSERT INTO"):
            raise ValueError("Generated response is not valid SQL INSERT statements.")
        return generated_sql
    except Exception as e:
        print(f"Error calling Gemini API for data generation: {e}")
        raise ValueError("Failed to generate data from Gemini API.")


# ✨ NEW: Helper function to generate questions using Gemini
async def generate_sql_questions(schema: str, sample_data_sql: str, num_questions: int) -> list[dict]:
    """
    Uses Gemini to generate SQL questions based on a schema and sample data,
    requesting a structured JSON output.
    """
    # Define the JSON schema for the expected response
    json_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "question_text": {"type": "string"},
                "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                "expected_sql": {"type": "string"}
            },
            "required": ["question_text", "difficulty", "expected_sql"]
        }
    }

    generation_config = GenerationConfig(
        response_mime_type="application/json",
        response_schema=json_schema
    )
    
    model = genai.GenerativeModel('gemini-2.5-flash', generation_config=generation_config)

    prompt = f"""
    You are an expert SQL instructor. Based on the following table schema and some sample data, please create {num_questions} SQL questions.

    The questions should be a good mix of difficulties: 'easy', 'medium', and 'hard'.
    For each question, provide the question text, its difficulty, and the ground-truth SQL query that correctly answers it.
    Your response MUST conform to the provided JSON schema.

    Table Schema:
    ```sql
    {schema}
    ```

    Sample Data (for context on the values in the table):
    ```sql
    {sample_data_sql[:2000]}
    ```
    """
    try:
        response = await model.generate_content_async(prompt)
        # The response.text should be a valid JSON string
        return json.loads(response.text)
    except Exception as e:
        print(f"Error calling Gemini API for question generation: {e}")
        raise ValueError("Failed to generate questions from Gemini API.")


@router.post("/upload")
async def upload_schema(payload: SchemaUpload, db: AsyncSession = Depends(get_db)):
    original_table_name = None
    unique_table_name = None
    test_id = None
    try:
        # 1. Extract the original table name from the provided schema
        schema_clean = payload.schema.strip()
        words = schema_clean.split()
        if len(words) >= 3 and words[0].upper() == "CREATE" and words[1].upper() == "TABLE":
            original_table_name = words[2].split('(')[0].strip().replace('`', '')
        
        if not original_table_name:
            raise HTTPException(status_code=400, detail="Could not parse table name from schema")

        # Start transaction
        async with db.begin():
            # 2. Insert a preliminary record into 'tests' to get a unique test_id.
            # We store the original schema and use the original table name as a placeholder.
            insert_test_sql = text("""
                INSERT INTO tests (name, description, schema_sql, table_name)
                VALUES (:name, :description, :schema_sql, :table_name)
                RETURNING id
            """)
            result = await db.execute(insert_test_sql, {
                "name": payload.test_name,
                "description": payload.description,
                "schema_sql": payload.schema,
                "table_name": original_table_name
            })
            test_id = result.scalar_one()

            # 3. ✨ NEW: Create a unique table name using the test_id
            unique_table_name = f"{original_table_name}_{test_id}"
            print(f"Generated unique table name: {unique_table_name}")

            # 4. ✨ NEW: Update the 'tests' record with the final, unique table name
            update_test_sql = text("UPDATE tests SET table_name = :unique_name WHERE id = :test_id")
            await db.execute(update_test_sql, {"unique_name": unique_table_name, "test_id": test_id})
            
            # 5. ✨ NEW: Modify the schema to use the unique table name
            modified_schema = schema_clean.replace(original_table_name, unique_table_name, 1)

            # 6. Execute the modified schema to create the uniquely named table
            await db.execute(text(modified_schema))
            
            # 7. Generate and insert data
            print(f"Generating data for table: {unique_table_name}...")
            # We pass the MODIFIED schema to Gemini so it generates inserts for the unique table name
            insert_data_sql = await generate_insert_statements(modified_schema, unique_table_name)
            
            if insert_data_sql:
                print(f"Inserting generated data into {unique_table_name}...")
                # Split into individual statements
                statements = [s.strip() for s in insert_data_sql.split(";") if s.strip()]
                for stmt in statements:
                    await db.execute(text(stmt))

            
            # 8. Generate questions based on the unique schema and data
            print(f"Generating {payload.num_questions} questions for test_id: {test_id}...")
            # Pass the MODIFIED schema here as well for correct question generation
            questions = await generate_sql_questions(modified_schema, insert_data_sql, payload.num_questions)
            
            # 9. Insert the generated questions into the database
            if questions:
                print(f"Inserting {len(questions)} generated questions into the database...")
                insert_question_sql = text("""
                    INSERT INTO questions (test_id, question_text, difficulty, expected_sql)
                    VALUES (:test_id, :question_text, :difficulty, :expected_sql)
                """)
                question_params = [
                    {
                        "test_id": test_id,
                        "question_text": q["question_text"],
                        "difficulty": q["difficulty"],
                        "expected_sql": q["expected_sql"]
                    } for q in questions
                ]
                await db.execute(insert_question_sql, question_params)

        return {
            "message": f"Schema created, data populated, and {len(questions)} questions generated successfully! ✅",
            "test_id": test_id,
            "table_name": unique_table_name # Return the new unique name
        }
        
    except Exception as e:
        # If an error occurs, the cleanup logic will use the unique name if it was generated
        if unique_table_name:
            try:
                await db.execute(text(f"DROP TABLE IF EXISTS {unique_table_name} CASCADE"))
                await db.commit()
            except Exception as drop_e:
                print(f"Failed to drop table {unique_table_name} on cleanup: {drop_e}")
                
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# Get all tests
# =====================================================
@router.get("/tests")
async def get_all_tests(db: AsyncSession = Depends(get_db)):
    """
    Fetch all tests from the database.
    """
    try:
        result = await db.execute(
            text("SELECT id, name, description, table_name, schema_sql FROM tests ORDER BY id DESC")
        )
        tests = result.fetchall()
        
        return {
            "tests": [
                {
                    "id": test[0],
                    "name": test[1],
                    "description": test[2],
                    "table_name": test[3],
                    "schema_sql": test[4]
                }
                for test in tests
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch tests: {str(e)}")

# =====================================================
# Get questions + schema + top 5 rows for a specific test
# =====================================================
@router.get("/tests/{test_id}/questions")
async def get_test_questions(test_id: int, db: AsyncSession = Depends(get_db)):
    """
    Fetch all questions for a specific test, 
    including schema and top 5 rows of data from the associated table.
    """
    try:
        # 1. Verify the test exists
        test_result = await db.execute(
            text("SELECT id, name, table_name, schema_sql FROM tests WHERE id = :test_id"),
            {"test_id": test_id}
        )
        test = test_result.fetchone()
        if not test:
            raise HTTPException(status_code=404, detail="Test not found")

        test_id_val, test_name, table_name, schema_sql = test

        # 2. Fetch questions for this test
        result = await db.execute(
            text("""
                SELECT id, question_text, difficulty, expected_sql 
                FROM questions 
                WHERE test_id = :test_id 
                ORDER BY id
            """),
            {"test_id": test_id_val}
        )
        questions = result.fetchall()

        # 3. Fetch schema of the actual table from database
        schema_result = await db.execute(
            text("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position
            """),
            {"table_name": table_name}
        )
        table_schema = schema_result.fetchall()

        # 4. Fetch top 5 rows from the actual table
        try:
            data_result = await db.execute(
                text(f"SELECT * FROM {table_name} LIMIT 5")
            )
            top_rows = [dict(row._mapping) for row in data_result.fetchall()]
        except Exception as e:
            top_rows = []
            print(f"Failed to fetch top rows for {table_name}: {e}")

        # 5. Return full response
        return {
            "test": {
                "id": test_id_val,
                "name": test_name,
                "table_name": table_name,
                "schema_sql": schema_sql,
                "table_schema": [
                    {"column_name": col[0], "data_type": col[1]}
                    for col in table_schema
                ],
                "top_rows": top_rows
            },
            "questions": [
                {
                    "id": q[0],
                    "question_text": q[1],
                    "difficulty": q[2],
                    "expected_sql": q[3]
                }
                for q in questions
            ]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch questions: {str(e)}")



# =====================================================
# NEW ENDPOINT: Validate a user's SQL answer
# =====================================================

@router.post("/validate")
async def validate_answer(
    test_id: int = Body(...),
    question_id: int = Body(...),
    user_sql: str = Body(...),
    db: AsyncSession = Depends(get_db)
):
    # 1. Fetch expected_sql
    result = await db.execute(
        text("SELECT expected_sql FROM questions WHERE id = :qid AND test_id = :tid"),
        {"qid": question_id, "tid": test_id}
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Question not found")
    expected_sql = row[0]

    try:
        # Expected SQL must be a SELECT
        expected_result = await db.execute(text(expected_sql))
        if not expected_result.returns_rows:
            raise HTTPException(status_code=400, detail="Expected SQL must be a SELECT")
        expected_rows = expected_result.fetchall()

        # User SQL
        user_result = await db.execute(text(user_sql))
        if not user_result.returns_rows:
            raise HTTPException(status_code=400, detail="Your SQL must be a SELECT query")
        user_rows = user_result.fetchall()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"SQL execution failed: {str(e)}")

    def normalize(rows):
        return sorted([tuple(r) for r in rows])

    return {
        "is_correct": normalize(expected_rows) == normalize(user_rows),
        "expected_output": normalize(expected_rows),
        "user_output": normalize(user_rows),
    }

@router.post("/submissions")
async def submit_quiz_result(
    submission: QuizSubmission,
    db: AsyncSession = Depends(get_db)
):
    """
    Store student quiz results in the database.
    USN will be capitalized and duplicates will be ignored.
    """
    try:
        # Capitalize the USN
        capitalized_usn = submission.student_usn.upper()
        
        # Check if submission already exists for this student and test
        check_sql = text("""
            SELECT id FROM quiz_submissions 
            WHERE student_usn = :usn AND test_id = :test_id
        """)
        result = await db.execute(
            check_sql, 
            {"usn": capitalized_usn, "test_id": submission.test_id}
        )
        existing_submission = result.fetchone()
        
        if existing_submission:
            return {
                "message": "Quiz already submitted for this student",
                "submission_id": existing_submission[0],
                "status": "duplicate"
            }
        
        # Insert new submission
        insert_sql = text("""
            INSERT INTO quiz_submissions 
            (student_name, student_usn, test_id, total_marks, max_marks, time_taken, submitted_at)
            VALUES (:name, :usn, :test_id, :total_marks, :max_marks, :time_taken, NOW())
            RETURNING id
        """)
        
        result = await db.execute(insert_sql, {
            "name": submission.student_name,
            "usn": capitalized_usn,
            "test_id": submission.test_id,
            "total_marks": submission.total_marks,
            "max_marks": submission.max_marks,
            "time_taken": submission.time_taken
        })
        
        submission_id = result.scalar_one()
        await db.commit()
        
        return {
            "message": "Quiz results stored successfully",
            "submission_id": submission_id,
            "status": "created"
        }
        
    except IntegrityError:
        await db.rollback()
        return {
            "message": "Quiz already submitted for this student",
            "status": "duplicate"
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to store submission: {str(e)}")

# Optional: Get submissions for a test
@router.get("/tests/{test_id}/submissions")
async def get_test_submissions(test_id: int, db: AsyncSession = Depends(get_db)):
    """
    Get all submissions for a specific test
    """
    try:
        result = await db.execute(
            text("""
                SELECT id, student_name, student_usn, total_marks, max_marks, 
                       time_taken, submitted_at
                FROM quiz_submissions 
                WHERE test_id = :test_id
                ORDER BY total_marks DESC, submitted_at DESC
            """),
            {"test_id": test_id}
        )
        
        submissions = result.fetchall()
        
        return {
            "submissions": [
                {
                    "id": sub[0],
                    "student_name": sub[1],
                    "student_usn": sub[2],
                    "total_marks": sub[3],
                    "max_marks": sub[4],
                    "time_taken": sub[5],
                    "submitted_at": sub[6].isoformat() if sub[6] else None
                }
                for sub in submissions
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch submissions: {str(e)}")