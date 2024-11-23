import json
import backoff
import openai
from icecream import ic

client = openai.OpenAI()

def get_structured_json_response_from_gpt(
        messages,
        response_format,
        model='gpt-4o-mini',
        temperature=0.5
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
            function_call = 'auto'
        )

    # Loading the response as a JSON object
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