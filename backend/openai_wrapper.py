import openai
import sqlite3
import uuid
from typing import List, Dict
import os
from dotenv import load_dotenv
load_dotenv(override=True)

class OpenAIClientWrapper:
    """I'm a poor student who can't afford too many queries haha"""
    
    def __init__(self, api_key: str, database_path: str = "prompts_cache.db"):
        self.client = openai.OpenAI(api_key=api_key)  
        self.db_path = database_path
        self._initialize_database()

    def _initialize_database(self):
        """Initializes the SQLite database for caching prompts."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prompts_cache (
            prompt_id UUID PRIMARY KEY,
            prompt TEXT UNIQUE,
            response TEXT
        )
        """)
        conn.commit()
        conn.close()

    def _get_cached_response(self, prompt: str) -> str:
        """Fetch a cached response from the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT response FROM prompts_cache WHERE prompt = ?", (prompt,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def _cache_response(self, prompt: str, response: str):
        """Cache a response in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO prompts_cache (prompt_id, prompt, response) VALUES (?, ?, ?)", 
                      (str(uuid.uuid4()), prompt, response))
        conn.commit()
        conn.close()

    def create(self, messages: List[Dict[str, str]], model: str = "gpt-4o-mini", temperature:float = 0.0, **kwargs) -> Dict:
        """
        Wrapper for the OpenAI chat completion API, with caching.
        """
        # Convert messages to a string for caching
        prompt_str = str(messages)
        
        # Check cache
        cached_response = self._get_cached_response(prompt_str)
        if cached_response:
            return {"choices": [{"message": {"content": cached_response}}]}

        # Make the OpenAI API call
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                **kwargs
            )
            content = response.choices[0].message.content
            
            # Cache the response
            self._cache_response(prompt_str, content)
            
            # Convert the response object to a dict for consistency
            return {
                "choices": [{
                    "message": {
                        "content": content
                    }
                }]
            }
        except Exception as e:
            raise Exception(f"Error making OpenAI API call: {str(e)}")

    def batch_generate(self, prompts: List[str], model: str = "gpt-4o-mini", temperature: float = 0.0) -> List[Dict]:
        """
        Batch generate responses for a list of prompts. Uses the cache when possible.
        """
        responses = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            cached_response = self._get_cached_response(str(messages))
            if cached_response:
                responses.append({"content": cached_response})
            else:
                response = self.create(messages=messages, model=model, temperature=temperature)
                content = response["choices"][0]["message"]["content"]
                responses.append({"content": content})
        return responses
    
    def generate(self, prompt: str) -> str:
        response = self.create(messages=[{"role": "user", "content": prompt}])["choices"][0]["message"]["content"]
        return response

if __name__ == "__main__":
    # Initialize the wrapper
    client = OpenAIClientWrapper(api_key=os.environ["OPENAI_API_KEY"])

    # Example: Single prompt with caching
    response = client.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "What is the capital of France?"}]
    )
    print("Single prompt response:", response["choices"][0]["message"]["content"])

    # Example: Batch generation
    prompts = ["What is the capital of France?", "What is the largest ocean on Earth?"]
    batch_responses = client.batch_generate(prompts)
    print("\nBatch responses:")
    for i, res in enumerate(batch_responses, 1):
        print(f"Response {i}:", res["content"])