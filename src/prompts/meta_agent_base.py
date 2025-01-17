import json
import sys
import time

sys.path.append("src")
from base import System


import os

from .utils import extract_class_code, extract_function_code
from .benchmark_prompts import benchmark_prompts


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

Agent_code = extract_class_code(f"{higher_directory}/base/system.py", "Agent")
get_structured_json_response_from_gpt_code = extract_function_code(
    f"{higher_directory}/chat/chat.py", "get_structured_json_response_from_gpt"
)

prompt_base = """# Overview
You are an expert machine learning researcher testing various agentic systems.
Your objective is to design building blocks such as prompts and control flows
within these systems to solve complex tasks.

[BENCHMARK_DESCRIPTION]

[SAFETY_DESCRIPTION]

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
    async def forward(self, task) -> str:
        \"""
        Placeholder method for processing task information.

        Args:
        - task (str): Task description.

        Returns:
        - Answer (str): Your FINAL Answer. Return as a string in the exact format as specified in the task.
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

# Documentation: Writing Forward Functions in Multi-Agent Framework
This documentation describes how to implement forward functions in your multi-agent framework, focusing on the interaction between Agents, Meetings, and Chats. Each forward function facilitates specific reasoning or task-solving approaches by coordinating these components effectively.

Framework Components
Agents: Autonomous entities with specific roles, goals, and configurations (e.g., temperature). They can participate in meetings and generate responses.
Meetings: Contextual containers where agents interact. Chats are added to meetings, and only agents in the meeting can "hear" the chats.
Chats: Messages exchanged in meetings. They capture the content generated by agents or instructions provided by the system.

## WRONG Implementation examples:
Here are some mistakes you may make:
## Anti-patterns to avoid:

1. DO NOT try to manually process agent outputs:
```python
# WRONG:
output = await agent.forward(...)
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

5. DO NOT output a dictionary with multiple keys:
```python
# WRONG:
async def forward(self, task):
    ...
    output:dict = await agent.forward(response_format={
    "thinking": "Your step by step reasoning",
    "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."
})
    return {"answer": output["answer"], "thinking": output["thinking"]}  # Don't output multiple keys
```

## CORRECT implementation patterns:

1. Proper agent creation and meeting setup:
```python
system = Agent(agent_name="system", temperature=0.7)
expert = Agent(agent_name="Expert", temperature=0.8)
meeting = Meeting(meeting_name="solving_task")
[agent.append(meeting) for agent in [system, expert]]
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
    "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."
})
return output["answer"]
```

4. Return the answer as a string:
```python
async def forward(self, task):
    output = await agent.forward(response_format={
    "thinking": "Your step by step reasoning",
    "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING."
})
    return output["answer"]

5. IMPORTANT: Always ensure the final answer is in the required format as stated in the task. This means the final agent must know about the task and its response format!


# Your task
You are deeply familiar with LLM prompting techniques and LLM agent works from the literature. Your goal is to maximize "fitness" by proposing interestingly new multi-agent systems.
Observe the discovered architectures carefully and think about what insights, lessons, or stepping stones can be learned from them.
Be creative to think about the next interesting architecture to try. You are encouraged to draw inspiration from related LLM agent papers or academic papers from other research areas.
Using the knowledge learned from the archive and the inspiration from academic literature to give the next interesting architecture.
THINK OUTSIDE THE BOX. Give a concise, powerful answer.
"""
prompt_base = prompt_base.replace("{{Agent_code}}", Agent_code)


def get_base_prompt_with_archive(args, session):

    # Only choose systems which haven't been validated yet (e.g. system_fitness=None)
    init_archive = (
        session.query(System)
        .filter_by(population_id=args.population_id, system_first_parent_id=None)
        .all()
    )

    archive = (
        session.query(System)
        .filter_by(population_id=args.population_id)
        .order_by(System.system_fitness.desc())
        .all()[:10]
    )

    # reverse the archive list to have the best solutions last
    archive = archive[::-1]

    archive_jsonl = []
    for system in init_archive + archive:
        if system.system_name not in [a["name"] for a in archive_jsonl]:
            archive_jsonl.append(
                {
                    "thought": system.system_thought_process,
                    "name": system.system_name,
                    "code": system.system_code,
                    "fitness": str(system.system_capability_ci_median),
                    "safety": str(system.system_safety_ci_median),
                }
            )

    archive_str = ",\n".join([json.dumps(sol, indent=4) for sol in archive_jsonl])
    archive_str = f"[{archive_str}]"
    prompt = prompt_base.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))
    prompt = prompt.replace(
        "[BENCHMARK_DESCRIPTION]", str(benchmark_prompts[args.benchmark])
    )
    if args.safety:
        prompt = prompt.replace(
            "[SAFETY_DESCRIPTION]", str(benchmark_prompts["salad_data"])
        )
    else:
        prompt = prompt.replace("[SAFETY_DESCRIPTION]", "")

    response_format = {
        "thought": "The first key should be (thought), and it should capture your thought process for designing the next function. In the thought section, first reason about what should be the next interesting agent to try, then describe your reasoning and the overall concept behind the agent design, and finally detail the implementation steps.",
        "name": "The second key (name) corresponds to the name of your next agent architecture.",
        "code": "Finally, the last key (code) corresponds to the exact forward() function in Python code that you would like to try. You must write a COMPLETE CODE in code: Your code will be part of the entire project, so please implement complete, reliable, reusable code snippets.",
    }

    return prompt, response_format
