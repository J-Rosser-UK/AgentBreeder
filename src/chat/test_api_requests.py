import time
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://localhost:8000/gpt"
RATE_LIMIT = 1000  # requests per second
NUM_REQUESTS = 100
WORKERS = 20  # Number of concurrent workers


def make_request(i: int):
    """
    Sends a single request to the server with a given index.
    """
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

    response = requests.post(URL, json=payload, timeout=None)
    response.raise_for_status()  # Raise an error for bad responses
    data = response.json()
    print(f"{i}: Received response for {question}: {data}")
    return i, data


def main():
    """
    Sends requests to the server concurrently with rate limiting.
    """
    send_pbar = tqdm(total=NUM_REQUESTS, desc="Requests Sent")
    recv_pbar = tqdm(total=NUM_REQUESTS, desc="Requests Received")

    results = []
    interval = 1 / RATE_LIMIT  # Time between requests in seconds
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(make_request, i): i for i in range(NUM_REQUESTS)}

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error on request {futures[future]}: {e}")
            finally:
                recv_pbar.update(1)

            # Enforce rate limit
            elapsed_time = time.time() - start_time
            if elapsed_time < interval * len(results):
                time.sleep(interval * len(results) - elapsed_time)

            send_pbar.update(1)

    send_pbar.close()
    recv_pbar.close()
    print("All tasks completed")
    return results


if __name__ == "__main__":
    main()
