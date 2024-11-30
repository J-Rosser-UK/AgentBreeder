import json
import backoff
import openai
from icecream import ic
import logging
from dotenv import load_dotenv
import os

load_dotenv(override=True)
client = openai.OpenAI()


# @backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt(
        msg,
        model,
        system_message,
        temperature=0.5
) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": msg},
        ],
        temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    # cost = response.usage.completion_tokens / 1000000 * 15 + response.usage.prompt_tokens / 1000000 * 5
    assert not json_dict is None
    return json_dict


# @backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_json_response_from_gpt_reflect(
        msg_list,
        model,
        temperature=0.8
) -> dict:
    response = client.chat.completions.create(
        model=model,
        messages=msg_list,
        temperature=temperature, max_tokens=4096, stop=None, response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    json_dict = json.loads(content)
    assert not json_dict is None
    return json_dict


# @backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_structured_json_response_from_gpt(
        messages,
        response_format,
        model='gpt-4o-mini',
        temperature=0.5,
        retry=0
) -> dict:
    
    properties = {}
    required = []
    for key, value in response_format.items():
        properties[key] = {"type": "string", "description": value}
        required.append(key)

    # Add "Please use the "get_structured_response" function to structure the response." to the final message
    messages.append({"role": "system", "content": "Please use the 'get_structured_response' function to structure the response."})

    response = client.chat.completions.create(
        model = model,
        temperature = temperature,
        messages=messages,
        functions = [
            {
                'name': 'get_structured_response',
                'description': 'Get structured response from GPT.',
                'parameters': {
                    'type': 'object',
                    'properties': properties,
                    'required': required
                }
            }
        ],
            function_call = { "name": "get_structured_response" }
        )

    # Loading the response as a JSON object
    if not response.choices[0].message.function_call and retry < 3:
        logging.warning("Retrying due to missing function call.")
        messages.append({"role": "system", "content": "YOU MUST use the 'get_structured_response' function to structure the response."})
        response = get_structured_json_response_from_gpt(messages, response_format, model, temperature, retry + 1)
    
    json_response = json.loads(response.choices[0].message.function_call.arguments)
    return json_response


if __name__ == "__main__":
    
    response = get_structured_json_response_from_gpt(
        messages=[
            {"role": "system", "content": "Please think step by step and then solve the task."},
            {"role": "user", "content": "What is the captial of France? A: Paris B: London C: Berlin D: Madrid."}
        ],
        response_format={"thinking": "Your step by step thinking.", "answer": "A single letter, A, B, C or D."},
    )
    print(response)