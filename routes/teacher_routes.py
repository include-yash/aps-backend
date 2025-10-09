from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError
from pydantic import BaseModel, Field
from typing import Optional
from utils.gemini import generate_insert_statements, generate_sql_questions
from database import get_db
import logging

# Setup logger
logger = logging.getLogger("teacher_routes")
logging.basicConfig(level=logging.INFO)

router = APIRouter(prefix="/teacher", tags=["Teacher"])

class SchemaUpload(BaseModel):
    teacher_id: Optional[int] = None
    test_name: str = Field(..., min_length=3)
    description: Optional[str] = None
    schema: str = Field(..., min_length=10)
    num_questions: int = Field(default=10, ge=1, le=50)  # Limit questions for production safety

# ----------------------------
# Upload Schema Endpoint
# ----------------------------
@router.post("/upload")
async def upload_schema(payload: SchemaUpload, db: AsyncSession = Depends(get_db)):
    """
    Upload a SQL schema, generate data and questions automatically using Gemini.
    """
    try:
        schema_clean = payload.schema.strip()
        words = schema_clean.split()
        if len(words) < 3 or words[0].upper() != "CREATE":
            logger.warning(f"Invalid schema submitted by teacher {payload.teacher_id}")
            raise HTTPException(status_code=400, detail="Invalid schema SQL. Must start with CREATE TABLE.")

        # Extract original table name
        original_table = words[2].split("(")[0].replace("`", "")
        logger.info(f"Uploading test '{payload.test_name}' with table '{original_table}'")

        async with db.begin():
            # Insert into test table
            res = await db.execute(
                text("""
                    INSERT INTO test (teacher_id, name, description, schema_sql, table_name)
                    VALUES (:tid, :name, :desc, :schema, :table)
                    RETURNING id
                """),
                {
                    "tid": payload.teacher_id,
                    "name": payload.test_name,
                    "desc": payload.description,
                    "schema": schema_clean,
                    "table": original_table
                }
            )
            test_id = res.scalar_one()

            # Ensure unique table name
            unique_table = f"{original_table}_{test_id}"
            await db.execute(text("UPDATE test SET table_name = :t WHERE id = :id"),
                             {"t": unique_table, "id": test_id})

            # Replace table name in schema and execute
            modified_schema = schema_clean.replace(original_table, unique_table, 1)
            await db.execute(text(modified_schema))

            # Generate insert statements
            try:
                insert_sql = await generate_insert_statements(modified_schema, unique_table)
            except Exception as e:
                logger.error(f"Failed to generate INSERT statements: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to generate data: {e}")

            # Execute insert statements safely
            for stmt in [s.strip() for s in insert_sql.split(";") if s.strip()]:
                try:
                    await db.execute(text(stmt))
                except SQLAlchemyError as e:
                    logger.error(f"Failed to execute INSERT statement: {stmt} | Error: {e}")
                    raise HTTPException(status_code=500, detail=f"Error executing generated data: {e}")

            # Generate SQL questions
            try:
                questions = await generate_sql_questions(modified_schema, insert_sql, payload.num_questions)
            except Exception as e:
                logger.error(f"Failed to generate SQL questions: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to generate questions: {e}")

            # Insert generated questions
            for q in questions:
                await db.execute(
                    text("""
                        INSERT INTO question (test_id, question_text, difficulty, expected_sql)
                        VALUES (:tid, :text, :diff, :sql)
                    """),
                    {"tid": test_id, "text": q["question_text"], "diff": q["difficulty"], "sql": q["expected_sql"]}
                )

        logger.info(f"Test uploaded successfully: {payload.test_name} (ID: {test_id})")
        return {"message": "Test uploaded successfully", "test_id": test_id, "table_name": unique_table}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during schema upload: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")
