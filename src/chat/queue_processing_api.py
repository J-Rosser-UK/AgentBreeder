from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import uuid
import time

app = FastAPI()


class Request(BaseModel):
    num1: float = 0.1
    num2: float = 0.2


# Global queue to hold (request_id, num1, num2)
request_queue = asyncio.Queue()

# Dictionary to hold results: request_id -> (result, event)
# The event will be set once the result is ready
pending_results = {}


# Background task that processes items from the queue
async def process_queue():
    while True:
        request_id, num1, num2 = await request_queue.get()
        # Simulate processing time and controlled rate
        await asyncio.sleep(1)  # process at 1 request every 0.01 seconds
        result = num1 + num2

        # Store the result and trigger the event so the waiting request can return
        res, event = pending_results[request_id]
        pending_results[request_id] = (result, event)
        event.set()


@app.on_event("startup")
async def startup_event():
    # Start the background worker task
    asyncio.create_task(process_queue())


@app.post("/add")
async def add(req: Request):
    # Generate a unique request_id
    request_id = str(uuid.uuid4())
    event = asyncio.Event()
    pending_results[request_id] = (None, event)

    # Add the request to the queue
    await request_queue.put((request_id, req.num1, req.num2))

    # Wait for the result to be processed
    await event.wait()

    # Retrieve the result and return it
    result, _ = pending_results[request_id]
    return {"request_id": request_id, "result": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8001)
