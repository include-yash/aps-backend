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
        logger.info(f"Received schema upload request for test: {payload.test_name}")
        logger.info(f"Teacher ID: {payload.teacher_id}")
        logger.info(f"Schema preview: {schema_clean[:200]}...")

        # Validate schema format
        words = schema_clean.split()
        if len(words) < 3 or words[0].upper() != "CREATE":
            logger.warning(f"Invalid schema submitted by teacher {payload.teacher_id}")
            raise HTTPException(status_code=400, detail="Invalid schema SQL. Must start with CREATE TABLE.")

        # Extract all original table names
        original_tables = []
        statements = [s.strip() for s in schema_clean.split(";") if s.strip()]
        
        for stmt in statements:
            stmt_words = stmt.split()
            if stmt_words[0].upper() == "CREATE" and stmt_words[1].upper() == "TABLE":
                # Extract table name, handling quotes
                table_name_part = stmt_words[2]
                original_name = table_name_part.replace("`", "").replace('"', '').replace("'", "")
                original_tables.append(original_name)
                logger.info(f"Found table: {original_name}")

        if not original_tables:
            raise HTTPException(status_code=400, detail="No tables found in schema.")

        async with db.begin():
            # Insert into test table
            logger.info("Inserting test record...")
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
                    "table": original_tables[0]
                }
            )
            test_id = res.scalar_one()
            logger.info(f"Test created with ID: {test_id}")

            # Create mapping from old table names to unique table names
            table_name_mapping = {orig: f"{orig}_{test_id}" for orig in original_tables}
            logger.info(f"Table name mapping: {table_name_mapping}")

            # Update test table's main table_name to first unique name
            await db.execute(
                text("UPDATE test SET table_name = :t WHERE id = :id"),
                {"t": table_name_mapping[original_tables[0]], "id": test_id}
            )

            # Execute each CREATE TABLE statement individually with updated names
            logger.info("Executing CREATE TABLE statements...")
            for stmt in statements:
                if not stmt.upper().startswith("CREATE TABLE"):
                    continue
                    
                # Extract and replace table name
                for old_name, new_name in table_name_mapping.items():
                    # Replace table name in CREATE TABLE
                    stmt = stmt.replace(f'"{old_name}"', f'"{new_name}"')
                    stmt = stmt.replace(f"`{old_name}`", f'"{new_name}"')
                    if f" {old_name} " in f" {stmt} ":
                        stmt = stmt.replace(f" {old_name} ", f' "{new_name}" ')
                    
                    # Replace references in foreign keys
                    stmt = stmt.replace(f'REFERENCES "{old_name}"', f'REFERENCES "{new_name}"')
                    stmt = stmt.replace(f"REFERENCES `{old_name}`", f'REFERENCES "{new_name}"')
                    if f"REFERENCES {old_name}" in stmt:
                        stmt = stmt.replace(f"REFERENCES {old_name}", f'REFERENCES "{new_name}"')

                logger.info(f"Executing: {stmt[:100]}...")
                try:
                    await db.execute(text(stmt))
                    logger.info("Statement executed successfully")
                except SQLAlchemyError as e:
                    logger.error(f"Failed to execute statement: {stmt} | Error: {e}")
                    raise HTTPException(status_code=500, detail=f"Error executing schema: {str(e)}")

            # Generate insert statements
            logger.info("Generating INSERT statements...")
            try:
                # Pass the original schema (without suffixes) to Gemini
                insert_sql = await generate_insert_statements(schema_clean, table_name_mapping)
                logger.info(f"Generated {len([s for s in insert_sql.split(';') if s.strip()])} INSERT statements")
            except Exception as e:
                logger.error(f"Failed to generate INSERT statements: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to generate data: {e}")

            # Execute insert statements safely
            logger.info("Executing INSERT statements...")
            insert_statements = [s.strip() for s in insert_sql.split(";") if s.strip()]
            for i, stmt in enumerate(insert_statements):
                try:
                    await db.execute(text(stmt))
                    if i % 5 == 0:  # Log every 5th statement
                        logger.info(f"Executed INSERT statement {i+1}/{len(insert_statements)}")
                except SQLAlchemyError as e:
                    logger.error(f"Failed to execute INSERT statement {i+1}: {stmt} | Error: {e}")
                    raise HTTPException(status_code=500, detail=f"Error executing generated data: {e}")

            # Generate SQL questions
            logger.info("Generating SQL questions...")
            try:
                questions = await generate_sql_questions(schema_clean, insert_sql, payload.num_questions)
                logger.info(f"Generated {len(questions)} questions")
            except Exception as e:
                logger.error(f"Failed to generate SQL questions: {e}")
                raise HTTPException(status_code=500, detail=f"Failed to generate questions: {e}")

            # Insert generated questions
            logger.info("Inserting questions into database...")
            for i, q in enumerate(questions):
                await db.execute(
                    text("""
                        INSERT INTO question (test_id, question_text, difficulty, expected_sql)
                        VALUES (:tid, :text, :diff, :sql)
                    """),
                    {"tid": test_id, "text": q["question_text"], "diff": q["difficulty"], "sql": q["expected_sql"]}
                )
            logger.info("Questions inserted successfully")

        logger.info(f"Test uploaded successfully: {payload.test_name} (ID: {test_id})")
        return {
            "message": "Test uploaded successfully", 
            "test_id": test_id, 
            "table_name": table_name_mapping[original_tables[0]]
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during schema upload: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    """
    Upload a SQL schema, generate data and questions automatically using Gemini.
    """
    try:
        schema_clean = payload.schema.strip()
        words = schema_clean.split()
        if len(words) < 3 or words[0].upper() != "CREATE":
            logger.warning(f"Invalid schema submitted by teacher {payload.teacher_id}")
            raise HTTPException(status_code=400, detail="Invalid schema SQL. Must start with CREATE TABLE.")

        # Extract all original table names
        original_tables = []
        for stmt in [s.strip() for s in schema_clean.split(";") if s.strip()]:
            stmt_words = stmt.split()
            if stmt_words[0].upper() == "CREATE" and stmt_words[1].upper() == "TABLE":
                original_name = stmt_words[2].replace("`", "")
                original_tables.append(original_name)

        if not original_tables:
            raise HTTPException(status_code=400, detail="No tables found in schema.")

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
                    "table": original_tables[0]
                }
            )
            test_id = res.scalar_one()

            # Create mapping from old table names to unique table names
            table_name_mapping = {orig: f"{orig}_{test_id}" for orig in original_tables}

            # Update test table's main table_name to first unique name
            await db.execute(
                text("UPDATE test SET table_name = :t WHERE id = :id"),
                {"t": table_name_mapping[original_tables[0]], "id": test_id}
            )

            # Execute each CREATE TABLE statement individually with updated names
            for stmt in [s.strip() for s in schema_clean.split(";") if s.strip()]:
                stmt_words = stmt.split()
                if stmt_words[0].upper() == "CREATE" and stmt_words[1].upper() == "TABLE":
                    original_name = stmt_words[2].replace("`", "")
                    unique_name = table_name_mapping[original_name]
                    stmt = stmt.replace(original_name, unique_name, 1)

                # Replace foreign key references
                for old_name, new_name in table_name_mapping.items():
                    stmt = stmt.replace(f"REFERENCES {old_name}", f"REFERENCES {new_name}")

                try:
                    await db.execute(text(stmt))
                except SQLAlchemyError as e:
                    logger.error(f"Failed to execute statement: {stmt} | Error: {e}")
                    raise HTTPException(status_code=500, detail=f"Error executing schema: {e}")

            # Generate insert statements
            try:
                insert_sql = await generate_insert_statements(schema_clean, table_name_mapping)
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
                questions = await generate_sql_questions(schema_clean, insert_sql, payload.num_questions)
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
        return {"message": "Test uploaded successfully", "test_id": test_id, "table_name": table_name_mapping[original_tables[0]]}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error during schema upload: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")