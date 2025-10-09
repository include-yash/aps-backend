import os
import json
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import google.generativeai as genai
import re
import logging
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv

load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY missing from .env")

genai.configure(api_key=GEMINI_API_KEY)

logger = logging.getLogger("gemini")
logging.basicConfig(level=logging.INFO)

# ThreadPoolExecutor for offloading JSON parsing to avoid blocking event loop
json_executor = ThreadPoolExecutor(max_workers=2)


async def async_json_loads(text: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(json_executor, json.loads, text)


async def _retry_async(func, retries=3, delay=2, backoff=2, *args, **kwargs):
    """Helper to retry async functions with exponential backoff"""
    current_delay = delay
    for attempt in range(1, retries + 1):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            if attempt == retries:
                raise
            logger.warning(f"Attempt {attempt} failed: {e}. Retrying in {current_delay}s...")
            await asyncio.sleep(current_delay)
            current_delay *= backoff


async def generate_insert_statements(schema: str, table_name_mapping: dict, num_rows: int = 15) -> str:
    """Generate realistic INSERT statements using Gemini with proper table name mapping."""
    # Input validation
    if not schema.strip():
        raise ValueError("Schema cannot be empty")
    if num_rows < 1 or num_rows > 1000:
        raise ValueError("num_rows must be between 1 and 1000")

    # Create a version of the schema with mapped table names for Gemini
    mapped_schema = schema
    for original_name, suffixed_name in table_name_mapping.items():
        # Replace table names in various quoting styles
        mapped_schema = mapped_schema.replace(f'"{original_name}"', f'"{suffixed_name}"')
        mapped_schema = mapped_schema.replace(f"`{original_name}`", f'"{suffixed_name}"')
        # Handle unquoted table names (add space context to avoid partial matches)
        mapped_schema = re.sub(rf'\s{re.escape(original_name)}\s', f' "{suffixed_name}" ', mapped_schema)

    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
    Generate {num_rows} valid PostgreSQL INSERT statements for the following SQL schema.
    Requirements:
    - Use the exact table names and column names as provided in the schema
    - Generate realistic, varied data that makes sense for each table
    - Ensure foreign key relationships are respected
    - Include all columns in each table
    - Use proper PostgreSQL syntax with double quotes for identifiers
    - Return ONLY the SQL INSERT statements separated by semicolons, no explanations

    Schema:
    {mapped_schema}

    Example format:
    INSERT INTO "table_name_123" ("id", "name") VALUES ('val1', 'Alice');
    INSERT INTO "table_name_123" ("id", "name") VALUES ('val2', 'Bob');
    """

    async def call_gemini():
        response = await model.generate_content_async(prompt)
        sql_text = response.text.strip()
        if "INSERT INTO" not in sql_text.upper():
            raise RuntimeError("Gemini did not return valid INSERT statements")
        return sql_text

    try:
        sql_text = await _retry_async(call_gemini)
        
        # Double-check that the generated SQL uses the correct table names
        # If Gemini somehow used original names, replace them
        for original_name, suffixed_name in table_name_mapping.items():
            if f'"{original_name}"' in sql_text or f"`{original_name}`" in sql_text:
                logger.warning(f"Replacing original table name '{original_name}' with '{suffixed_name}' in generated INSERTs")
                sql_text = sql_text.replace(f'"{original_name}"', f'"{suffixed_name}"')
                sql_text = sql_text.replace(f"`{original_name}`", f'"{suffixed_name}"')
                # Handle unquoted table names in INSERT statements
                sql_text = re.sub(rf'INSERT INTO\s+{re.escape(original_name)}\s', f'INSERT INTO "{suffixed_name}" ', sql_text)

        logger.info("Successfully generated INSERT statements with proper table names")
        return sql_text
    except Exception as e:
        logger.error(f"Failed to generate INSERT statements: {e}")
        raise RuntimeError("INSERT generation failed") from e

async def generate_sql_questions(schema: str, sample_data_sql: str, num_questions: int = 10) -> list[dict]:
    """Generate structured SQL questions using Gemini based on schema and sample data."""
    # Input validation
    if not schema.strip():
        raise ValueError("Schema cannot be empty")
    if not sample_data_sql.strip():
        raise ValueError("Sample data cannot be empty")
    if num_questions < 1 or num_questions > 50:
        raise ValueError("num_questions must be between 1 and 50")

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

    model = genai.GenerativeModel("gemini-2.5-flash", generation_config=generation_config)
    prompt = f"""
    Create {num_questions} diverse SQL questions based on this schema and sample data.
    Return valid JSON matching the given schema.

    Schema:
    {schema}

    Sample Data:
    {sample_data_sql[:2000]}
    """

    async def call_gemini_questions():
        response = await model.generate_content_async(prompt)
        data = await async_json_loads(response.text)
        if not isinstance(data, list):
            raise RuntimeError("Gemini returned invalid question format")
        return data

    try:
        questions = await _retry_async(call_gemini_questions)
        logger.info("Successfully generated SQL questions")
        return questions
    except Exception as e:
        logger.error(f"Failed to generate SQL questions: {e}")
        raise RuntimeError("Question generation failed") from e
