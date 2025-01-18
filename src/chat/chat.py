import json
import backoff
import openai
from icecream import ic
import logging
from dotenv import load_dotenv
import os
import asyncio

load_dotenv(override=True)
# client = openai.OpenAI()
import httpx


client = httpx.AsyncClient()
URL = "http://localhost:8001/gpt"

# https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py


async def get_structured_json_response_from_gpt(
    messages, response_format, model="gpt-4o-mini", temperature=0.5, retry=0
) -> dict:
    payload = {
        "messages": messages,
        "response_format": response_format,
        "model": model,
        "temperature": temperature,
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(URL, json=payload, timeout=None)
        data = response.json().get("result")
        return data
    except httpx.RequestError as e:
        logging.error(f"An error occurred while requesting {e.request.url!r}.")
        return {}
    except json.JSONDecodeError as e:
        logging.error(f"JSON decode error: {e}")
        return {}


async def main():
    response = await get_structured_json_response_from_gpt(
        messages=[
            {
                "role": "system",
                "content": "Please think step by step and then solve the task.",
            },
            {
                "role": "user",
                "content": "What is the captial of France? A: Paris B: London C: Berlin D: Madrid.",
            },
        ],
        response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D.",
        },
    )
    print(response)


if __name__ == "__main__":

    asyncio.run(main())
