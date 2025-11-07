from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from typing import List, Tuple, Dict, Optional
import logging
import re
from time import sleep
from mapping import MappingOutput
from pydantic import ValidationError
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate and initialize OpenAI client
#_api_key = os.getenv("OPENAI_API_KEY")
_api_key = st.secrets["OPENAI_API_KEY"]
if not _api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")
client = OpenAI(api_key=_api_key)



def load_prompt(path: str) -> str:
    """
    Loads a text prompt template from a file.

    Args:
        path (str): The file path to the prompt template.

    Returns:
        str: The full text content of the prompt file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If the file cannot be read for any other reason.
    """
    if not path:
        raise ValueError("Path cannot be empty")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            if not content.strip():
                logger.warning(f"Prompt file is empty: {path}")
            return content
    except FileNotFoundError:
        logger.error(f"Prompt file not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load prompt from {path}: {e}")
        raise IOError(f"Failed to load prompt from {path}: {e}") from e


def build_prompt(
    prompt_template: str,
    user_text: str,
    top_matches: List[Tuple[str, float]], 
    queries_dict: Dict[str, Dict]
) -> str:
    """
    Builds the full LLM prompt by injecting user question and top-matching query descriptions.

    Args:
        prompt_template (str): The prompt string with placeholders.
        user_text (str): The original user question.
        top_matches (List[Tuple[str, float]]): Top query keys with similarity scores.
        queries_dict (dict): Full content of queries.json.

    Returns:
        str: The formatted prompt ready to send to LLM.
        
    Raises:
        ValueError: If no valid query descriptions found or inputs are invalid.
    """
    # Input validation
    if not prompt_template or not prompt_template.strip():
        raise ValueError("prompt_template cannot be empty")
    if not user_text or not user_text.strip():
        raise ValueError("user_text cannot be empty")
    if not top_matches:
        raise ValueError("top_matches cannot be empty")
    if not queries_dict:
        raise ValueError("queries_dict cannot be empty")
    
    desc = []
    
    for k, _ in top_matches:
        try:
            desc.append({
                "key": k,
                "description": queries_dict[k]["description"]
            })
        except KeyError:
            logger.warning(f"Query key '{k}' not found in queries_dict.")
            continue
    
    if not desc:
        raise ValueError("No valid query descriptions found for given top_matches.")

    try:
        prompt = prompt_template.format(
            user_text=user_text,
            desc=json.dumps(desc, indent=2)
        )
    except KeyError as e:
        raise ValueError(f"Missing placeholder in prompt_template: {e}") from e
    
    return prompt
    

def call_llm(prompt: str, model: Optional[str] = None, temperature: float = 0.0) -> str:
    """
    Sends the prompt to the OpenAI LLM and returns the raw text response.

    Args:
        prompt (str): The prompt to send.
        model (str, optional): The model to use. Defaults to 'gpt-4o'.
        temperature (float, optional): Sampling temperature. Defaults to 0.0.

    Returns:
        str: Raw response text from the model.

    Raises:
        ValueError: If prompt is empty.
        RuntimeError: If the LLM call fails or response is empty.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
    
    try:
        logger.debug(f"Calling LLM with model={model}, temperature={temperature}")
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        if not response.choices:
            raise RuntimeError("LLM returned no choices")
        
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned empty response")
            
        return content.strip()
        
    except Exception as e:
        logger.exception(f"LLM call failed with model {model}")
        raise RuntimeError(f"LLM call failed: {e}") from e

def parse_response(raw_text: str) -> dict:
    """
    Extracts and parses JSON from a raw LLM response.
    
    Args:
        raw_text (str): Raw text returned by the LLM.

    Returns:
        dict: Parsed JSON object.

    Raises:
        ValueError: If raw_text is empty.
        RuntimeError: If valid JSON cannot be extracted.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text cannot be empty")
    
    # First attempt - direct parse
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        logger.warning("Direct JSON parse failed. Trying regex fallback.")
    
    # Second attempt - regex extract with improved pattern
    # Match outermost JSON object, avoiding greedy matching
    match = re.search(r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}', raw_text, re.DOTALL)
    if not match:
        logger.error(f"No JSON object found in LLM response. Response: {raw_text[:200]}")
        raise RuntimeError("No JSON object found in LLM response.")
    
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.error(f"Regex-extracted JSON is still invalid: {e}")
        raise RuntimeError(f"Failed to parse JSON: {e}") from e
        
        
def validate_llm_response(data: dict) -> dict:
    """
    Validates the parsed LLM response using Pydantic schema.

    Args:
        data (dict): JSON dict from LLM.

    Returns:
        dict: Validated and normalized version.

    Raises:
        ValueError: If structure or types are wrong.
    """
    try:
        validated = MappingOutput(**data)
        return validated.model_dump()
    except ValidationError as e:
        error_messages = []
        for err in e.errors():
            loc = " → ".join(str(x) for x in err["loc"])
            msg = err["msg"]
            error_messages.append(f"{loc}: {msg}")
        full_error = " | ".join(error_messages)
        raise ValueError(f"Invalid LLM response: {full_error}")


def validate_and_retry(
    prompt: str, 
    max_retries: int = 3, 
    retry_delay: float = 1.0
) -> dict:
    """
    Calls LLM with a prompt, parses and validates the result.
    Retries with feedback if validation fails.

    Args:
        prompt (str): Initial LLM prompt.
        max_retries (int): Maximum retry attempts. Defaults to 3.
        retry_delay (float): Delay between retries in seconds. Defaults to 1.0.

    Returns:
        dict: Validated query mapping with query_key and params.

    Raises:
        ValueError: If prompt is empty or max_retries is invalid.
        RuntimeError: If validation fails after max_retries.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt cannot be empty")
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    
    original_prompt = prompt
    last_error = None
    
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Attempt {attempt}/{max_retries}")
            raw = call_llm(prompt)
            parsed = parse_response(raw)
            validated = validate_llm_response(parsed)
            logger.info(f"✅ Valid LLM response received on attempt {attempt}")
            return validated
            
        except Exception as e:
            last_error = e
            logger.warning(f"⚠️ Attempt {attempt}/{max_retries} failed: {e}")
            
            if attempt < max_retries:
                # Provide feedback to LLM for next attempt
                prompt = (
                    original_prompt +
                    f"\n\n⚠️ Your previous response could not be validated.\n"
                    f"Reason: {e}\n"
                    f"Please respond with valid JSON in this format:\n"
                    f'{{"query_key": "...", "params": {{"..."}}}}'
                )
                logger.debug(f"Retrying after {retry_delay}s delay")
                sleep(retry_delay)
            else:
                logger.error(f"❌ All {max_retries} attempts failed")
    
    raise RuntimeError(
        f"❌ LLM failed after {max_retries} attempts.\n"
        f"Last error: {last_error}"
    ) from last_error
            