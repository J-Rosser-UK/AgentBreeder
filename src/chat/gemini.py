import json
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import os
import backoff
from typing_extensions import TypedDict
from rich import print
from chat import get_structured_json_response_from_gpt
import google.generativeai as genai
from google.generativeai.protos import Tool, FunctionDeclaration, Schema, Type
from google.protobuf.json_format import MessageToDict

# Load environment variables
load_dotenv(override=True)

# Configure Gemini
genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def validate_response(response: str, expected_fields: set) -> dict:
    """
    Validates and ensures all expected fields are present in the response.

    Args:
        response (str): JSON response from Gemini
        expected_fields (set): Set of field names that should be present

    Returns:
        dict: Validated and complete JSON response
    """
    # Parse the response
    if isinstance(response, str):
        parsed = json.loads(response)
    else:
        parsed = response

    # Ensure all expected fields are present
    for field in expected_fields:
        if field not in parsed:
            raise Exception(f"Field '{field}' is missing from the response")

    return parsed


# @backoff.on_exception(backoff.expo, Exception)
def get_structured_json_response_from_gemini(
    messages: list[dict],
    response_format: dict,
    model_name: str = "gemini-1.5-flash",
    temperature: float = 0.5,
    retry: int = 0,
) -> dict:
    """
    Gets a structured JSON response from Gemini with guaranteed field presence.

    Args:
        messages (list): List of message dictionaries with "role" and "content".
        response_format (dict): A dictionary defining the expected JSON fields and their descriptions.
        model_name (str): The Gemini model to use.
        temperature (float): Sampling temperature.
        retry (int): Retry count for handling exceptions.

    Returns:
        dict: A parsed JSON response matching the specified format with all fields present.
    """
    # Create more explicit field descriptions with examples
    field_descriptions = []
    for key, value in response_format.items():
        field_descriptions.append(
            f"- {key}: {value if value else 'No description provided'}"
        )

    # Enhanced system prompt with explicit instructions about field requirements
    system_message = (
        "You are an AI assistant. Your response MUST be a valid JSON object containing ALL of the following fields:\n"
        + "\n".join(field_descriptions)
        + "\n\nCritical requirements:"
        + "\n1. Include ALL specified fields in your response"
        + "\n2. Use null for empty/unknown values, but never omit fields"
        + "\n3. Ensure the response is valid JSON"
        + "\n4. Do not include any additional fields"
        + "\n\nExample format:"
        + "\n{"
        + "\n  "
        + ",\n  ".join(f'"{key}": <value or null>' for key in response_format.keys())
        + "\n}"
    )

    messages.append({"role": "system", "content": system_message})

    # Create TypedDict for response validation
    StructuredResponse = TypedDict(
        "StructuredResponse", {key: str for key in response_format.keys()}
    )

    try:
        model = genai.GenerativeModel(model_name)
        result = model.generate_content(
            "\n".join(
                message["content"] for message in messages if message["role"] == "user"
            ),
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                response_schema=StructuredResponse,
                temperature=temperature,
            ),
        )

        # Validate and ensure all fields are present
        validated_response = validate_response(result.text, set(response_format.keys()))

        return validated_response

    except Exception as e:
        # logging.warning(f"Error during Gemini generation, switching to GPT: {e}")
        return get_structured_json_response_from_gpt(
            messages, response_format, "gpt-4o-mini", temperature, retry
        )


# Example usage
if __name__ == "__main__":
    response_format = {
        # "thinking": "Thinking style of the AI",
        "recipe_name": "Name of the recipe",
        "ingredients": "List of ingredients required for the recipe",
    }
    messages = [
        {
            "role": "system",
            "content": 'You are an AI assistant. Your response MUST be a valid JSON object containing ALL of the following fields:\n- thinking: Thinking style of the AI\n- recipe_name: Name of the recipe\n- ingredients: List of ingredients required for the recipe\n\nCritical requirements:\n1. Include ALL specified fields in your response\n2. Use null for empty/unknown values, but never omit fields\n3. Ensure the response is valid JSON\n4. Do not include any additional fields\n\nExample format:\n{\n  "thinking": "<value or null>",\n  "recipe_name": "<value or null>",\n  "ingredients": "<value or null>"\n}',
        },
        {"role": "user", "content": "Write a recipe for a delicious chocolate cake."},
    ]

    try:
        structured_response = get_structured_json_response_from_gemini(
            messages, response_format
        )
        print(structured_response)
    except Exception as e:
        print(f"Error: {e}")
