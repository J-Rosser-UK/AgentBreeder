from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import uuid
import time
import json
import logging
import os
from typing import Dict, Any
import openai
import tiktoken
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv(override=True)
client = openai.OpenAI()

# ----------------------------------
# CONFIGURATION & GLOBAL VARIABLES
# ----------------------------------
MAX_REQUESTS_PER_MINUTE = 10000  # adjust as needed
MAX_TOKENS_PER_MINUTE = 10000000  # adjust as needed
MAX_ATTEMPTS = 3
TOKEN_ENCODING_NAME = "cl100k_base"
MODEL = "gpt-4o-mini"  # adjust as needed
N = MAX_REQUESTS_PER_MINUTE // 60 - 10  # We'll dequeue one item every 1/N seconds

logging.basicConfig(level=logging.INFO)

# We'll maintain a queue of requests to process.
request_queue = asyncio.Queue()

# We'll maintain a dictionary of results keyed by request_id.
results: Dict[str, Any] = {}

# We'll maintain a dictionary to hold pending results: request_id -> (result, event)
pending_results: Dict[str, Any] = {}

# To track requests per second, we'll keep counters
calls_completed_in_current_second = 0
last_log_time = time.time()


def count_tokens(messages):
    """Count tokens for chat completion requests using tiktoken."""
    encoding = tiktoken.get_encoding(TOKEN_ENCODING_NAME)
    num_tokens = 0
    for message in messages:
        # 4 tokens for role/name delim and message, plus tokens for content
        num_tokens += 4
        for val in message.values():
            num_tokens += len(encoding.encode(val))
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


def call_openai_sync(messages, response_format, model=MODEL, temperature=0.5, retry=0):
    """Synchronous call to OpenAI, used in threadpool executor."""
    properties = {}
    required = []
    for key, value in response_format.items():
        properties[key] = {"type": "string", "description": value}
        required.append(key)

    # Add "Please use the "get_structured_response" function to structure the response." to the final message
    messages.append(
        {
            "role": "system",
            "content": "Please use the 'get_structured_response' function to structure the response.",
        }
    )

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
        max_tokens=2000,
        functions=[
            {
                "name": "get_structured_response",
                "description": "Get structured response from GPT.",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            }
        ],
        timeout=60,
        function_call={"name": "get_structured_response"},
    )

    # Loading the response as a JSON object
    if not response.choices[0].message.function_call and retry < 3:
        logging.warning("Retrying due to missing function call.")
        messages.append(
            {
                "role": "system",
                "content": "YOU MUST use the 'get_structured_response' function to structure the response.",
            }
        )
        response = call_openai_sync(
            messages, response_format, model, temperature, retry + 1
        )

    json_response = json.loads(response.choices[0].message.function_call.arguments)
    return json_response


app = FastAPI()


class GPTRequest(BaseModel):
    messages: list = [{"role": "user", "content": "Hello!"}]
    response_format: dict = {"response": "A response."}
    model: str = MODEL
    temperature: float = 0.5


@app.post("/gpt")
async def gpt_endpoint(req: GPTRequest):
    token_consumption = count_tokens(req.messages)
    req_id = str(time.time()) + "_" + str(id(req))
    logging.info(
        f"Queueing request {req_id} with token consumption {token_consumption}"
    )

    event = asyncio.Event()
    pending_results[req_id] = (None, event)

    await request_queue.put(
        (
            req_id,
            req.messages,
            req.response_format,
            req.model,
            req.temperature,
            MAX_ATTEMPTS,
            token_consumption,
        )
    )

    await event.wait()
    result, _ = pending_results[req_id]
    return {"request_id": req_id, "result": result}


# Global executor for concurrent calls
executor = ThreadPoolExecutor(max_workers=N)


def future_callback(fut, req_id):
    """Callback when a future completes."""
    global calls_completed_in_current_second
    exc = fut.exception()
    if exc:
        # If there's an exception, you could handle retries or store error
        logging.error(f"Error in processing {req_id}: {exc}")
        result = {"error": str(exc)}
    else:
        result = fut.result()
    # Set the result and trigger the event so the waiting request can return
    res, event = pending_results[req_id]
    pending_results[req_id] = (result, event)
    event.set()
    calls_completed_in_current_second += 1


async def process_scheduler():
    """A scheduler task that tries to dequeue 1 item every 1/N seconds."""
    while True:
        try:
            item = request_queue.get_nowait()
        except asyncio.QueueEmpty:
            # If queue is empty, just sleep for 1/N second
            await asyncio.sleep(1.0 / N)
            continue

        (
            req_id,
            messages,
            response_format,
            model,
            temperature,
            attempts_left,
            token_consumption,
        ) = item
        # Submit the call to executor immediately, no waiting
        fut = executor.submit(
            call_openai_sync, messages, response_format, model, temperature
        )
        fut.add_done_callback(lambda f, r=req_id: future_callback(f, r))

        # Sleep 1/N second before trying to dequeue next item
        await asyncio.sleep(1.0 / N)


async def log_rate():
    """Logs the rate of openai calls per second every second."""
    global calls_completed_in_current_second
    while True:
        await asyncio.sleep(1)
        # Log how many calls completed in the last second
        logging.info(f"OpenAI calls per second: {calls_completed_in_current_second}")
        calls_completed_in_current_second = 0


@app.on_event("startup")
async def startup_event():
    # Start scheduler and logger tasks
    asyncio.create_task(process_scheduler())
    asyncio.create_task(log_rate())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
