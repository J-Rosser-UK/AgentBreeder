#!/usr/bin/env python
"""
Example: Single-file Custom ModelAPI and Model registration + usage.
"""

import asyncio
from typing import Any

# -- Imports from Inspect AI
from inspect_ai.model._model import ModelAPI, Model, get_model
from inspect_ai.model._generate_config import GenerateConfig
from inspect_ai.model._chat_message import ChatMessage
from inspect_ai.model._model_output import ModelOutput
from inspect_ai.tool import ToolInfo, ToolChoice
from inspect_ai.model._registry import modelapi


@modelapi(name="agentbreeder")
class CustomModelAPI(ModelAPI):
    """
    Custom ModelAPI provider that you can reference as "agentbreeder/<model_name>".
    By decorating with @modelapi("agentbreeder"), we ensure it is known to Inspect's registry.
    """

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_vars: list[str] = [],
        config: GenerateConfig = GenerateConfig(),
        **model_args: Any,
    ) -> None:
        # IMPORTANT: call super() so Inspect sets up the registry info correctly
        super().__init__(model_name, base_url, api_key, api_key_vars, config)

    async def generate(
        self,
        input: list[ChatMessage],
        tools: list[ToolInfo],
        tool_choice: ToolChoice,
        config: GenerateConfig,
    ) -> ModelOutput:
        """
        Minimal example: just respond with a constant string.
        You could, of course, wrap an API call here to an external LLM, a local model, etc.
        """

        # Example: Just echo back that this is "my agentbreeder model's" response
        response_text = f"[CustomModelAPI] Hello from {self.model_name}!"
        return ModelOutput.from_content(model=self.model_name, content=response_text)


@modelapi(name="agentbreeder")
def agentbreeder():

    return CustomModelAPI


class CustomModel(Model):
    """
    Example of a agentbreeder Model that overrides Model’s behavior if you wish
    (though typically you only need a agentbreeder ModelAPI).
    """

    async def generate(
        self,
        input: str | list[ChatMessage] = None,
        tools: list[ToolInfo] = [],
        tool_choice: ToolChoice | None = None,
        config: GenerateConfig = GenerateConfig(),
        cache: bool = False,
    ) -> ModelOutput:
        """
        Minimal override. You could do more advanced orchestration here if you like.
        For demonstration, we always return "Hello from CustomModel!"
        """
        # Just return a trivial output
        return ModelOutput.from_content(
            model=self.api.model_name, content="Hello from CustomModel!"
        )


def main():
    """
    Minimal example usage in __main__.
    We'll do two calls:
      1. Use the Inspect get_model() to reference our CustomModelAPI.
      2. Show how you might directly instantiate CustomModel if you prefer.
    """

    # 1) Use the registry-based approach: "agentbreeder/gpt-3.5-turbo"
    #    - "agentbreeder" is the provider name we gave in @modelapi(name="agentbreeder")
    #    - "gpt-3.5-turbo" is the model_name we pass to CustomModelAPI
    model_via_registry = get_model("agentbreeder/gpt-3.5-turbo")
    output1 = asyncio.run(model_via_registry.generate("Hello from registry approach"))
    print("===== Output from CustomModelAPI (registry) =====")
    print(output1.choices[0].message.content, "\n")

    # 2) Use the fully agentbreeder class approach:
    #    Instantiate the ModelAPI yourself, then pass it to your agentbreeder Model class.
    agentbreeder_api = CustomModelAPI(
        model_name="my-manual-model",
        config=GenerateConfig(max_tokens=42),  # Example config
    )
    model_manual = CustomModel(api=agentbreeder_api, config=GenerateConfig())
    output2 = asyncio.run(model_manual.generate("Hello from direct instantiation"))
    print("===== Output from CustomModel (manual) =====")
    print(output2.choices[0].message.content, "\n")


if __name__ == "__main__":
    main()
