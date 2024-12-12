import asyncio
import json
import logging
import os
import time
from typing import Dict, Any
from icecream import ic
import openai
import tiktoken
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(override=True)
openai.api_key = os.getenv("OPENAI_API_KEY")

# ----------------------------------
# CONFIGURATION & GLOBAL VARIABLES
# ----------------------------------
MAX_REQUESTS_PER_MINUTE = 5000  # adjust as needed
MAX_TOKENS_PER_MINUTE = 2000000  # adjust as needed
MAX_ATTEMPTS = 3
TOKEN_ENCODING_NAME = "cl100k_base"
MODEL = "gpt-4o-mini"  # adjust as needed

# We'll maintain a queue of requests to process.
request_queue = asyncio.Queue()
# We'll also maintain a dictionary of results keyed by request_id.
results: Dict[str, Any] = {}
# We'll maintain a dictionary to track the status and metadata of each request
requests_info: Dict[str, Any] = {}

# Status tracker counters
num_rate_limit_errors = 0
time_of_last_rate_limit_error = 0

# Rate limit state
available_request_capacity = MAX_REQUESTS_PER_MINUTE
available_token_capacity = MAX_TOKENS_PER_MINUTE
last_update_time = time.time()


# ----------------------------------
# UTILITY FUNCTIONS
# ----------------------------------
def count_tokens(messages):
    """Count tokens for chat completion requests using tiktoken."""
    encoding = tiktoken.get_encoding(TOKEN_ENCODING_NAME)
    # Approximate counting method for chat messages:
    num_tokens = 0
    for message in messages:
        # 4 tokens for role/name delim and message, plus tokens for content
        num_tokens += 4
        for val in message.values():
            num_tokens += len(encoding.encode(val))
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens


async def call_openai(messages, response_format, model=MODEL, temperature=0.5):
    """Call OpenAI with function calls and return structured JSON."""
    properties = {}
    required = []
    for key, value in response_format.items():
        properties[key] = {"type": "string", "description": value}
        required.append(key)

    # Add system message requesting structured response
    final_messages = messages + [
        {
            "role": "system",
            "content": "Please use the 'get_structured_response' function to structure the response.",
        }
    ]

    logging.info(f"Calling OpenAI with messages: {final_messages}")
    response = openai.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=final_messages,
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
        function_call={"name": "get_structured_response"},
    )
    json_response = json.loads(response.choices[0].message.function_call.arguments)
    return json_response


async def process_requests():
    """Background task that processes requests from the queue, applying rate limits."""
    global available_request_capacity, available_token_capacity, last_update_time
    global num_rate_limit_errors, time_of_last_rate_limit_error

    seconds_to_pause_after_rate_limit_error = 15
    while True:
        # If no requests, just wait
        if request_queue.empty():
            await asyncio.sleep(0.01)
            continue

        current_time = time.time()
        seconds_since_update = current_time - last_update_time

        # Refill capacities
        available_request_capacity = min(
            MAX_REQUESTS_PER_MINUTE,
            available_request_capacity
            + (MAX_REQUESTS_PER_MINUTE * seconds_since_update / 60.0),
        )
        available_token_capacity = min(
            MAX_TOKENS_PER_MINUTE,
            available_token_capacity
            + (MAX_TOKENS_PER_MINUTE * seconds_since_update / 60.0),
        )
        last_update_time = current_time

        # If rate-limited error recently, pause
        seconds_since_rate_limit_error = current_time - time_of_last_rate_limit_error
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            await asyncio.sleep(
                seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
            )
            continue

        (
            req_id,
            messages,
            response_format,
            model,
            temperature,
            attempts_left,
            token_consumption,
        ) = await request_queue.get()

        # Update status to processing
        requests_info[req_id]["status"] = "processing"
        requests_info[req_id]["processing_start_time"] = time.time()

        if (
            available_request_capacity >= 1
            and available_token_capacity >= token_consumption
        ):
            # Consume capacity
            available_request_capacity -= 1
            available_token_capacity -= token_consumption

            # Attempt API call
            error = None
            try:
                result = await call_openai(
                    messages, response_format, model, temperature
                )
                # Success
                results[req_id] = {"status": "success", "data": result}
            except openai.RateLimitError as e:
                error = e
                num_rate_limit_errors += 1
                time_of_last_rate_limit_error = time.time()
            except Exception as e:
                error = e

            if error:
                if attempts_left > 1:
                    # Retry
                    requests_info[req_id]["status"] = "in_queue"
                    requests_info[req_id]["last_error"] = str(error)
                    await request_queue.put(
                        (
                            req_id,
                            messages,
                            response_format,
                            model,
                            temperature,
                            attempts_left - 1,
                            token_consumption,
                        )
                    )
                else:
                    # Fail after all attempts
                    requests_info[req_id]["status"] = "error"
                    requests_info[req_id]["error"] = str(error)
                    results[req_id] = {"status": "error", "error": str(error)}
            else:
                requests_info[req_id]["status"] = "completed"
                requests_info[req_id]["completed_time"] = time.time()
        else:
            # Not enough capacity right now, put request back and wait a bit
            requests_info[req_id]["status"] = "in_queue"
            await request_queue.put(
                (
                    req_id,
                    messages,
                    response_format,
                    model,
                    temperature,
                    attempts_left,
                    token_consumption,
                )
            )
            await asyncio.sleep(0.01)


# ----------------------------------
# FASTAPI APP
# ----------------------------------
app = FastAPI(debug=True)


class GPTRequest(BaseModel):
    messages: list = [{"role": "user", "content": "Hello!"}]
    response_format: dict = {"response": "A response."}
    model: str = MODEL
    temperature: float = 0.5


@app.on_event("startup")
async def startup_event():
    # Start background task
    asyncio.create_task(process_requests())


@app.post("/gpt")
async def gpt_endpoint(req: GPTRequest):
    token_consumption = count_tokens(req.messages)
    req_id = str(time.time()) + "_" + str(id(req))
    # Put request in queue
    logging.info(
        f"Queueing request {req_id} with token consumption {token_consumption}"
    )
    print(f"Queueing request {req_id} with token consumption {token_consumption}")

    # Store request info
    requests_info[req_id] = {
        "status": "in_queue",
        "enqueued_time": time.time(),
        "token_count": token_consumption,
        "model": req.model,
        "temperature": req.temperature,
    }

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

    # Wait for result
    while True:
        print("TRUE")
        if req_id in results:
            result = results.pop(req_id)
            if result["status"] == "success":
                return result["data"]
            else:
                raise HTTPException(
                    status_code=500, detail=result.get("error", "Unknown error")
                )
        await asyncio.sleep(0.01)


@app.get("/status")
async def status_endpoint():
    # Return a list of all requests and their statuses
    # Include how long they have been in queue/processed
    now = time.time()
    response_list = []
    for req_id, info in requests_info.items():
        status = info["status"]
        enqueued_time = info["enqueued_time"]
        time_in_system = now - enqueued_time

        # If completed or error, we can show total time
        # If still in queue or processing, show current waiting time
        entry = {
            "request_id": req_id,
            "status": status,
            "token_count": info.get("token_count"),
            "time_in_system_seconds": time_in_system,
            "model": info.get("model"),
            "temperature": info.get("temperature"),
        }

        if status == "completed":
            entry["completed_time"] = info.get("completed_time", None)
            if entry["completed_time"]:
                entry["total_processing_time"] = entry["completed_time"] - enqueued_time
        elif status == "error":
            entry["error"] = info.get("error")

        response_list.append(entry)

    return response_list


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
