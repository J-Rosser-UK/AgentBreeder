import json
from utils import extract_class_code, extract_function_code
import os
EXAMPLE = {
    "thought": "**Insights:**\nYour insights on what should be the next interesting agent.\n**Overall Idea:**\nyour reasoning and the overall concept behind the agent design.\n**Implementation:**\ndescribe the implementation step by step.",
    "name": "Name of your proposed agent",
    "code": """def forward(self, task):
    # Your code here
    return answer
"""
}

COT = {
    "thought": "By encouraging the LLM to think step by step rather than directly outputting an answer, chain-of-thought reasoning enables complex problem-solving through intermediate steps. This practice improves the model's ability to handle tasks that require deeper reasoning and provides insight into its decision-making process.",
    "name": "Chain-of-Thought",
    "code": """def forward(self, task: str) -> str:
    # Create a system agent to provide instructions
    system = Agent(
        agent_name='system',
        temperature=0.8
    )
    
    # Create the Chain-of-Thought agent
    cot_agent = Agent(
        agent_name='Chain-of-Thought Agent',
        temperature=0.7
    )
    
    # Setup meeting
    meeting = Meeting(meeting_name="chain-of-thought")
    meeting.agents.extend([system, cot_agent])
    
    # Add system instruction
    meeting.chats.append(
        Chat(
            agent=system, 
            content=f"Please think step by step and then solve the task: {task}"
        )
    )
    
    # Get response from COT agent
    output = cot_agent.forward(
        response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        }
    )
    
    # Record the agent's response in the meeting
    meeting.chats.append(
        Chat(
            agent=cot_agent, 
            content=output["thinking"]
        )
    )
    
    return output["answer"]
"""
}

COT_SC = {"thought": "While an LLM can arrive at the correct answer, its reasoning may vary. By repeatedly asking the same question with high temperature settings, we can generate different reasoning paths. We then combine multiple answers from these Chain-of-Thought (CoT) agents to produce a more accurate final answer through ensembling.",
          "name": "Self-Consistency with Chain-of-Thought",
          "code": """def forward(self, task: str) -> str:
    # Create a system agent to provide instructions
    system = Agent(
        agent_name='system',
        temperature=0.8
    )
    
    # Create multiple CoT agents with higher temperature for varied reasoning
    N = 5  # Number of CoT agents
    cot_agents = [
        Agent(
            agent_name=f'Chain-of-Thought Agent {i}',
            temperature=0.8
        ) for i in range(N)
    ]
    
    # Setup meeting
    meeting = Meeting(meeting_name="self-consistency")
    meeting.agents.extend([system] + cot_agents)
    
    # Collect answers from all agents
    possible_answers = []
    for i in range(N):
        # Add system instruction
        meeting.chats.append(
            Chat(
                agent=system, 
                content=f"Please think step by step and then solve the task: {task}"
            )
        )
        
        # Get response from current COT agent
        output = cot_agents[i].forward(
            response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            }
        )
        
        # Record the agent's response
        meeting.chats.append(
            Chat(
                agent=cot_agents[i], 
                content=output["thinking"]
            )
        )
        
        possible_answers.append(output["answer"])
    
    # Select the most common answer through majority voting
    from collections import Counter
    
    final_answer = Counter(possible_answers).most_common(1)[0][0]
    return final_answer
"""
          }

Reflexion = {
    "thought": "To enhance its performance, an LLM can iteratively improve its answer based on feedback. By reflecting on its previous attempts and incorporating feedback, the model can refine its reasoning and provide a more accurate solution.",
    "name": "Self-Refine (Reflexion)",
    "code": """def forward(self, task: str) -> str:
    # Create system and agent instances
    system = Agent(
        agent_name='system',
        temperature=0.8
    )
    
    cot_agent = Agent(
        agent_name='Chain-of-Thought Agent',
        temperature=0.7
    )
    
    critic_agent = Agent(
        agent_name='Critic Agent',
        temperature=0.6
    )
    
    # Setup meeting
    meeting = Meeting(meeting_name="reflexion")
    meeting.agents.extend([system, cot_agent, critic_agent])
    
    N_max = 5  # Maximum number of attempts
    
    # Initial attempt
    meeting.chats.append(
        Chat(
            agent=system, 
            content=f"Please think step by step and then solve the task: {task}"
        )
    )
    
    output = cot_agent.forward(
        response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        }
    )
    
    meeting.chats.append(
        Chat(
            agent=cot_agent, 
            content=output["thinking"]
        )
    )
    
    # Refinement loop
    for i in range(N_max):
        # Get feedback from critic
        meeting.chats.append(
            Chat(
                agent=system, 
                content="Please review the answer above and criticize where it might be wrong. If you are absolutely sure it is correct, output 'CORRECT'."
            )
        )
        
        critic_output = critic_agent.forward(
            response_format={
                "feedback": "Your detailed feedback.",
                "correct": "Either 'CORRECT' or 'INCORRECT'"
            }
        )
        
        meeting.chats.append(
            Chat(
                agent=critic_agent, 
                content=critic_output["feedback"]
            )
        )
        
        if critic_output["correct"] == "CORRECT":
            break
        
        # Reflect and refine
        meeting.chats.append(
            Chat(
                agent=system, 
                content=f"Given the feedback above, carefully consider where you could go wrong in your latest attempt. Using these insights, try to solve the task better: {task}"
            )
        )
        
        output = cot_agent.forward(
            response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            }
        )
        
        meeting.chats.append(
            Chat(
                agent=cot_agent, 
                content=output["thinking"]
            )
        )
    
    return output["answer"]
"""
}

LLM_debate = {
    "thought": "By letting different LLMs debate with each other, we can leverage their diverse perspectives to find better solutions for tasks.",
    "name": "LLM Debate",
    "code": """def forward(self, task: str) -> str:

    # Create a system agent to provide instructions
    system = Agent(agent_name = 'system', temperature=0.8)

    # Initialize debate agents with different roles and a moderate temperature for varied reasoning
    debate_agents = [Agent(
        agent_name=name,
        temperature=0.8
    ) for name in ['Biology Expert', 'Physics Expert', 'Science Generalist']]

    # Instruction for final decision-making based on all debates and solutions
    final_decision_agent = Agent(agent_name = 'Final Decision Agent',temperature=0.1)
    
    # Setup a single meeting for the debate
    meeting = Meeting(meeting_name="debate")

    # Ensure all agents are part of the meeting
    [meeting.agents.append(agent) for agent in debate_agents]
    meeting.agents.append(system)
    meeting.agents.append(final_decision_agent)

    max_round = 2 # Maximum number of debate rounds

    # Perform debate rounds
    for r in range(max_round):
        for i in range(len(debate_agents)):
            if r == 0 and i == 0:
                meeting.chats.append(Chat(agent=system, content=f"Please think step by step and then solve the task: {task}"))
                output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "A single letter, A, B, C or D."})
                
            else:
                meeting.chats.append(Chat(agent=system, content=f"Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. Reminder, the task is: {task}"))
                output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "A single letter, A, B, C or D."})

            meeting.chats.append(Chat(agent=debate_agents[i], content=output["thinking"]+output["response"]))

    # Make the final decision based on all debate results and solutions
    meeting.chats.append(Chat(agent=system, content="Given all the above thinking and answers, reason over them carefully and provide a final answer."))
    output = final_decision_agent.forward(response_format = {"thinking": "Your step by step thinking.", "answer": "A single letter, A, B, C or D."})
    
    return output["answer"]
"""
}

Take_a_step_back = {"thought": "Let LLM first think about the principles involved in solving this task which could be helpful. By understanding the underlying principles, the model can better reason through the problem and provide a more accurate solution.",
                    "name": "Step-back Abstraction",
                    "code": """def forward(self, task: str) -> str:
    # Create agents
    system = Agent(agent_name='system', temperature=0.8)
    principle_agent = Agent(agent_name='Principle Agent', temperature=0.8)
    cot_agent = Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
    
    # Setup meeting
    meeting = Meeting(meeting_name="step_back_meeting")
    meeting.agents.extend([system, principle_agent, cot_agent])
    
    # First get the principles involved
    meeting.chats.append(Chat(
        agent=system,
        content="What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
    ))
    
    principle_output = principle_agent.forward(response_format={
        "thinking": "Your step by step thinking about the principles.",
        "principles": "List and explanation of the principles involved."
    })
    
    meeting.chats.append(Chat(
        agent=principle_agent,
        content=principle_output["thinking"] + principle_output["principles"]
    ))
    
    # Now solve using the principles
    meeting.chats.append(Chat(
        agent=system,
        content=f"Given the question and the involved principles above, think step by step and then solve the task: {task}"
    ))
    
    final_output = cot_agent.forward(response_format={
        "thinking": "Your step by step thinking.",
        "answer": "A single letter, A, B, C or D."
    })
    
    return final_output["answer"]
"""
                    }

QD = {"thought": "Similar to Quality-Diversity methods, let LLM generate multiple diverse interesting solutions could help. By encouraging the model to explore different reasoning paths, we can increase the chances of finding the best solution.",
      "name": "Quality-Diversity",
      "code": """def forward(self, task: str) -> str:
    # Create agents
    system = Agent(agent_name='system', temperature=0.8)
    cot_agent = Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
    final_decision_agent = Agent(agent_name='Final Decision Agent', temperature=0.1)
    
    # Setup meeting
    meeting = Meeting(meeting_name="quality_diversity_meeting")
    meeting.agents.extend([system, cot_agent, final_decision_agent])
    
    N_max = 3  # Maximum number of attempts
    
    # Initial attempt
    meeting.chats.append(Chat(
        agent=system,
        content=f"Please think step by step and then solve the task: {task}"
    ))
    
    output = cot_agent.forward(response_format={
        "thinking": "Your step by step thinking.",
        "answer": "A single letter, A, B, C or D."
    })
    
    meeting.chats.append(Chat(
        agent=cot_agent,
        content=output["thinking"] + output["answer"]
    ))
    
    # Generate diverse solutions
    for i in range(N_max):
        meeting.chats.append(Chat(
            agent=system,
            content=f"Given previous attempts, try to come up with another interesting way to solve the task: {task}"
        ))
        
        output = cot_agent.forward(response_format={
            "thinking": "Your step by step thinking with a new approach.",
            "answer": "A single letter, A, B, C or D."
        })
        
        meeting.chats.append(Chat(
            agent=cot_agent,
            content=output["thinking"] + output["answer"]
        ))
    
    # Make final decision
    meeting.chats.append(Chat(
        agent=system,
        content="Given all the above solutions, reason over them carefully and provide a final answer."
    ))
    
    final_output = final_decision_agent.forward(response_format={
        "thinking": "Your step by step thinking comparing all solutions.",
        "answer": "A single letter, A, B, C or D."
    })
    
    return final_output["answer"]
"""
      }

Role_Assignment = {"thought": "Similar to Auto-GPT and expert prompting, we can use dynamic control flow in the design to let the agent decide what expert we should use.",
                   "name": "Dynamic Assignment of Roles",
                   "code": """def forward(self, task: str) -> str:
    # Create agents
    system = Agent(agent_name='system', temperature=0.8)
    routing_agent = Agent(agent_name='Routing Agent', temperature=0.8)
    
    expert_agents = {
        'physics': Agent(agent_name='Physics Expert', temperature=0.8),
        'chemistry': Agent(agent_name='Chemistry Expert', temperature=0.8),
        'biology': Agent(agent_name='Biology Expert', temperature=0.8),
        'general': Agent(agent_name='Science Generalist', temperature=0.8)
    }
    
    # Setup meeting
    meeting = Meeting(meeting_name="role_assignment_meeting")
    meeting.agents.extend([system, routing_agent] + list(expert_agents.values()))
    
    # Route the task
    meeting.chats.append(Chat(
        agent=system,
        content="Given the task, please choose an Expert to answer the question. Choose from: Physics, Chemistry, Biology Expert, or Science Generalist."
    ))
    
    routing_output = routing_agent.forward(response_format={
        "choice": "One of: physics, chemistry, biology, or general"
    })
    
    # Select expert based on routing decision
    expert_choice = routing_output["choice"].lower()
    if expert_choice not in expert_agents:
        expert_choice = 'general'
        
    selected_expert = expert_agents[expert_choice]
    
    # Get answer from selected expert
    meeting.chats.append(Chat(
        agent=system,
        content=f"Please think step by step and then solve the task: {task}"
    ))
    
    expert_output = selected_expert.forward(response_format={
        "thinking": "Your step by step thinking.",
        "answer": "A single letter, A, B, C or D."
    })
    
    return expert_output["answer"]
"""
                   }

system_prompt = """You are a helpful assistant. Make sure to return in a WELL-FORMED JSON object."""

current_directory = os.path.dirname(os.path.abspath(__file__))

Agent_code = extract_class_code(f"{current_directory}/agent.py", "Agent")
get_json_response_from_gpt_code = extract_function_code(f"{current_directory}/chat.py", "get_json_response_from_gpt")
get_json_response_from_gpt_reflect_code = extract_function_code(f"{current_directory}/chat.py", "get_json_response_from_gpt_reflect")


base = """# Overview
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

{{get_json_response_from_gpt_code}}

{{get_json_response_from_gpt_reflect_code}}

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
THINK OUTSIDE THE BOX.
"""
base = base.replace("{{Agent_code}}", Agent_code)
base = base.replace("{{get_json_response_from_gpt_code}}", get_json_response_from_gpt_code)
base = base.replace("{{get_json_response_from_gpt_reflect_code}}", get_json_response_from_gpt_reflect_code)


Reflexion_prompt_1 = f""""[EXAMPLE]Carefully review the proposed new architecture and reflect on the following points:"

1. **Interestingness**: Assess whether your proposed architecture is interesting or innovative compared to existing methods in the archive. If you determine that the proposed architecture is not interesting, suggest a new architecture that addresses these shortcomings. 
- Make sure to check the difference between the proposed architecture and previous attempts.
- Compare the proposal and the architectures in the archive CAREFULLY, including their actual differences in the implementation.
- Decide whether the current architecture is innovative.
- USE CRITICAL THINKING!

2. **Implementation Mistakes**: Identify any mistakes you may have made in the implementation. Review the code carefully, debug any issues you find, and provide a corrected version. REMEMBER checking "## WRONG Implementation examples" in the prompt.

3. **Improvement**: Based on the proposed architecture, suggest improvements in the detailed implementation that could increase its performance or effectiveness. In this step, focus on refining and optimizing the existing implementation without altering the overall design framework, except if you want to propose a different architecture if the current is not interesting.
- Observe carefully about whether the implementation is actually doing what it is supposed to do.
- Check if there is redundant code or unnecessary steps in the implementation. Replace them with effective implementation.
- Try to avoid the implementation being too similar to the previous agent.

And then, you need to improve or revise the implementation, or implement the new proposed architecture based on the reflection.

Your response should be organized as follows:

"reflection": Provide your thoughts on the interestingness of the architecture, identify any mistakes in the implementation, and suggest improvements.

"thought": Revise your previous proposal or propose a new architecture if necessary, using the same format as the example response.

"name": Provide a name for the revised or new architecture. (Don't put words like "new" or "improved" in the name.)

"code": Provide the corrected code or an improved implementation. Make sure you actually implement your fix and improvement in this code.
"""

Reflexion_prompt_2 = """Using the tips in "## WRONG Implementation examples" section, revise the code further.
Your response should be organized as follows:
Put your new reflection thinking in "reflection". Repeat the previous "thought" and "name", and update the corrected version of the code in "code".
"""


def get_init_archive():
    return [COT, COT_SC, Reflexion, LLM_debate, Take_a_step_back, QD, Role_Assignment]


def get_prompt(current_archive, adaptive=False):
    archive_str = ",\n".join([json.dumps(sol) for sol in current_archive])
    archive_str = f"[{archive_str}]"
    prompt = base.replace("[ARCHIVE]", archive_str)
    prompt = prompt.replace("[EXAMPLE]", json.dumps(EXAMPLE))

    return system_prompt, prompt


def get_reflexion_prompt(prev_example):
    prev_example_str = "Here is the previous agent you tried:\n" + json.dumps(prev_example) + "\n\n"
    r1 = Reflexion_prompt_1.replace("[EXAMPLE]", prev_example_str) if prev_example else Reflexion_prompt_1.replace("[EXAMPLE]", "")
    return r1, Reflexion_prompt_2
