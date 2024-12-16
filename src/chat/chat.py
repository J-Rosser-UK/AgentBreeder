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
URL = "http://localhost:8000/gpt"

# https://github.com/openai/openai-cookbook/blob/main/examples/api_request_parallel_processor.py


async def get_structured_json_response_from_gpt(
    messages, response_format, model="gpt-4o-mini", temperature=0.5, retry=0
) -> dict:
    # logging.info("Getting structured JSON response from GPT.")
    payload = {
        "messages": messages,
        "response_format": response_format,
        "model": model,
        "temperature": temperature,
    }

    response = await client.post(URL, json=payload, timeout=None)

    data = response.json()["result"]
    # print(data)

    # logging.info(data)

    return data


# def get_structured_json_response_from_gpt(
#     messages, response_format, model="gpt-4o-mini", temperature=0.5, retry=0
# ) -> dict:

#     properties = {}
#     required = []
#     for key, value in response_format.items():
#         properties[key] = {"type": "string", "description": value}
#         required.append(key)

#     # Add "Please use the "get_structured_response" function to structure the response." to the final message
#     messages.append(
#         {
#             "role": "system",
#             "content": "Please use the 'get_structured_response' function to structure the response.",
#         }
#     )

#     response = client.chat.completions.create(
#         model=model,
#         temperature=temperature,
#         messages=messages,
#         functions=[
#             {
#                 "name": "get_structured_response",
#                 "description": "Get structured response from GPT.",
#                 "parameters": {
#                     "type": "object",
#                     "properties": properties,
#                     "required": required,
#                 },
#             }
#         ],
#         function_call={"name": "get_structured_response"},
#     )

#     # Loading the response as a JSON object
#     if not response.choices[0].message.function_call and retry < 3:
#         logging.warning("Retrying due to missing function call.")
#         messages.append(
#             {
#                 "role": "system",
#                 "content": "YOU MUST use the 'get_structured_response' function to structure the response.",
#             }
#         )
#         response = get_structured_json_response_from_gpt(
#             messages, response_format, model, temperature, retry + 1
#         )

#     json_response = json.loads(response.choices[0].message.function_call.arguments)
#     return json_response


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
