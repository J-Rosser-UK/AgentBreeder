import json
import backoff
import openai
from icecream import ic

client = openai.OpenAI()


@backoff.on_exception(backoff.expo, openai.RateLimitError)
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


@backoff.on_exception(backoff.expo, openai.RateLimitError)
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


from pydantic import BaseModel, create_model, Field


def dict_to_class(class_name: str, format: dict) -> BaseModel:
    """
    Dynamically creates a Pydantic BaseModel subclass based on the provided dictionary.

    :param format: A dictionary where keys are field names and values are their types.
    :param class_name: The name of the dynamically created class.
    :return: A new Pydantic BaseModel subclass with the specified fields.
    """
    # Prepare the fields for create_model.
    # Pydantic expects each field to be a tuple of (type, default), where default can be ...
    fields = {field_name: (field_type, ...) for field_name, field_type in format.items()}

    # Create the model dynamically
    dynamic_model = create_model(class_name, **fields)

    return dynamic_model


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