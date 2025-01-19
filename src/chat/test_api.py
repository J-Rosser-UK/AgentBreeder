import asyncio
import httpx
from tqdm import tqdm

URL = "http://localhost:8000/gpt"
RATE_LIMIT = 1000  # requests per second
NUM_REQUESTS = 100


async def rate_limiter(semaphore: asyncio.Semaphore):
    # This task periodically releases a semaphore, allowing a new request
    # to start at a regular interval
    while True:
        await asyncio.sleep(1 / RATE_LIMIT)
        semaphore.release()


async def make_request(
    client: httpx.AsyncClient, i: int, semaphore: asyncio.Semaphore, send_pbar
):
    # Wait until the rate_limiter releases a slot
    async with semaphore:
        # Once we acquire the semaphore, it means we're now allowed to send a request
        question = f"What's {i} + {i+1}?"
        payload = {
            "messages": [{"role": "user", "content": question}],
            "response_format": {
                "thinking": "Your step by step thinking.",
                "answer": "A single number.",
            },
            "model": "gpt-4o-mini",
            "temperature": 0.5,
        }

        # Update the "sending" progress bar just before sending
        send_pbar.update(1)
        response = await client.post(URL, json=payload, timeout=None)

        response.raise_for_status()  # Will raise if the request failed
        data = response.json()
        print(f"{i}: Received response for {question}: {data}")
        return i, data


async def main():
    indices = range(NUM_REQUESTS)
    semaphore = asyncio.Semaphore(0)

    # Start the rate limiter
    asyncio.create_task(rate_limiter(semaphore))

    # Create two progress bars: one for sending, one for receiving
    send_pbar = tqdm(total=NUM_REQUESTS, desc="Requests Sent")
    recv_pbar = tqdm(total=NUM_REQUESTS, desc="Requests Received")

    async with httpx.AsyncClient() as client:
        # Create all tasks upfront
        tasks = [make_request(client, i, semaphore, send_pbar) for i in indices]

        # As tasks complete, update the received progress bar
        results = []
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            recv_pbar.update(1)

        send_pbar.close()
        recv_pbar.close()
        print("All tasks completed")
        return results


if __name__ == "__main__":
    asyncio.run(main())
