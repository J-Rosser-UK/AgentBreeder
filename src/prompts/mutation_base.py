import json
from .utils import extract_class_code, extract_function_code
from .initial_population import (
    COT,
    COT_SC,
    Reflexion,
    LLM_debate,
    Take_a_step_back,
    QD,
    Role_Assignment,
)
import os

EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting agent.\n**Overall Idea:**\nyour reasoning and the overall concept behind the agent design.\n**Implementation:**\ndescribe the implementation step by step.",
    "name": "Name of your proposed agent",
    "code": """async def forward(self, task):
    # Your code here
    return answer
""",
}

current_directory = os.path.dirname(os.path.abspath(__file__))
higher_directory = os.path.dirname(current_directory)

Agent_code = extract_class_code(f"{higher_directory}/base/tables.py", "Agent")
get_structured_json_response_from_gpt_code = extract_function_code(
    f"{higher_directory}/chat/chat.py", "get_structured_json_response_from_gpt"
)

prompt_base = """# Overview
You are an expert machine learning researcher testing various agentic systems. Your objective is to design building blocks such as prompts and control flows within these systems to solve complex tasks. Your aim is to design an optimal agent performing well on the MMLU (Massive Multitask Language Understanding) benchmark, a challenging evaluation that assesses a model's ability to answer questions across a wide range of subjects and difficulty levels. It includes subjects from STEM, social sciences, humanities, and more.

## An example question from MMLU:

Answer the following multiple choice question.

The constellation ... is a bright W-shaped constellation in the northern sky.

(A) Centaurus
(B) Cygnus
(C) Cassiopeia
(D) Cepheus

# The utility code:

```python
from collections import namedtuple
from typing import Union
import numpy as np
import json

import openai
import backoff
from utils import random_id

# Initialize the OpenAI client
client = openai.OpenAI()

{{get_structured_json_response_from_gpt_code}}

{{Agent_code}}

class AgentArchitecture:
    \"""
    Fill in your code here.
    \"""
    def forward(self, task) -> str:
        \"""
        Placeholder method for processing task information.
        
        Args:
        - task (str): Task description.
        
        Returns:
        - Answer (str): Your FINAL Answer. Return as a string.
        \"""
        pass
```
# Discovered architecture archive
Here is the archive of the discovered architectures:

[ARCHIVE]

The fitness value is the median and 95% Bootstrap Confidence Interval of the correct rate on a validation question set. Your GOAL is to maximize the "fitness".

# Output Instruction and Example:
The first key should be ("thought"), and it should capture your thought process for designing the next function. In the "thought" section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.
The second key ("name") corresponds to the name of your next agent architecture. 
Finally, the last key ("code") corresponds to the exact “forward()” function in Python code that you would like to try. You must write a COMPLETE CODE in "code": Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.

Here is an example of the output format for the next agent architecture:

[EXAMPLE]

You must use the exact function interface used above. You need to specify the instruction, input information, and the required output fields for various LLM agents to do their specific part of the architecture. 
Also, it could be helpful to set the LLMs role and temperature to further control the LLMs response. Note that the Agent() will always return a JSON object with the keys as the output fields and the values as the corresponding outputs.
DO NOT FORGET the task input to LLM if you think it is needed, otherwise LLM will not know about the task.

## WRONG Implementation examples:
Here are some mistakes you may make:
## Anti-patterns to avoid:

1. DO NOT try to manually process agent outputs:
```python
# WRONG:
output = agent.forward(...)
processed = process_output(output["thinking"])  # Don't process outputs manually
```

2. DO NOT use print statements or error handling that returns error messages:
```python
# WRONG:
print("Debug:", output)  # No print statements
if not output:
    return "Error"  # No error messages as returns
```

3. DO NOT try to join or aggregate outputs manually:
```python
# WRONG:
all_outputs = []
for agent in agents:
    output = agent.forward(...)
    all_outputs.append(output["content"])  # Don't manually aggregate
combined = "\\n".join(all_outputs)  # Don't manually join
```

4. DO NOT extract or manipulate response content directly:
```python
# WRONG:
output = agent.forward(...)
if output["answer"] == "A":  # Don't inspect content directly
    return "A"
```

## CORRECT implementation patterns:

1. Proper agent creation and meeting setup:
```python
system = Agent(agent_name="system", temperature=0.7)
expert = Agent(agent_name="Expert", temperature=0.8)
meeting = Meeting(meeting_name="solving_task")
meeting.agents.extend([system, expert])
```

2. Proper chat message addition:
```python
meeting.chats.append(Chat(
    agent=system,
    content=f"Please solve this task: \{task\}"
))
```

3. Proper response format usage:
```python
output = agent.forward(response_format={
    "thinking": "Your step by step reasoning",
    "answer": "Single letter A, B, C, or D"
})
return output["answer"]
```

# Your task
You are deeply familiar with LLM prompting techniques and LLM agent works from the literature. Your goal is to maximize "fitness" by proposing interestingly new agents. 
Observe the discovered architectures carefully and think about what insights, lessons, or stepping stones can be learned from them.
Be creative to think about the next interesting architecture to try. You are encouraged to draw inspiration from related LLM agent papers or academic papers from other research areas.
Using the knowledge learned from the archive and the inspiration from academic literature to give the next interesting architecture.
THINK OUTSIDE THE BOX. Give a concise, powerful answer.
"""
prompt_base = prompt_base.replace("{{Agent_code}}", Agent_code)
# prompt_base = prompt_base.replace(
#     "{{get_structured_json_response_from_gpt_code}}",
#     get_structured_json_response_from_gpt_code,
# )


def get_init_archive():
    return [COT, COT_SC, Reflexion, LLM_debate, Take_a_step_back, QD, Role_Assignment]


def get_base_prompt(adaptive=False):
    archive_str = ",\n".join([json.dumps(sol) for sol in get_init_archive()])
    archive_str = f"[{archive_str}]"
    prompt = prompt_base.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))

    response_format = {
        "thought": "The first key should be (thought), and it should capture your thought process for designing the next function. In the thought section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.",
        "name": "The second key (name) corresponds to the name of your next agent architecture.",
        "code": "Finally, the last key (code) corresponds to the exact forward() function in Python code that you would like to try. You must write a COMPLETE CODE in code: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.",
    }

    return prompt, response_format
