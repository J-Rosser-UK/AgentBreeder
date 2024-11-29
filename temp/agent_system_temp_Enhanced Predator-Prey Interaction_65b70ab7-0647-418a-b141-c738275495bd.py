import random
import numpy as np
import pandas

from base import Agent, Meeting, Chat, Wrapper

from sqlalchemy.orm import Session

class AgentSystem:
    def __init__(self, session: Session):
        self.Agent = Wrapper(Agent, session)
        self.Meeting = Wrapper(Meeting, session)
        self.Chat = Wrapper(Chat, session)
        self.session = session

    def forward(self, task: str) -> str:\n    # Create agents\n    system = self.Agent(agent_name='system', temperature=0.8)\n    predator_agents = {\n        'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),\n        'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),\n        'biology': self.Agent(agent_name='Biology Expert', temperature=0.8)\n    }\n    prey_agents = [self.Agent(agent_name='Prey Agent', temperature=0.7) for _ in range(2)]\n    \n    # Setup meeting\n    meeting = self.Meeting(meeting_name="predator_prey_meeting")\n    meeting.agents.extend([system] + list(predator_agents.values()) + prey_agents)\n    \n    # Each predator presents their reasoning\n    predator_outputs = []  # Store predator outputs for evaluation\n    for predator in predator_agents.values():\n        meeting.chats.append(self.Chat(\n            agent=system,\n            content=f"{predator.agent_name}, please think step by step and present your reasoning for the task: {task}"\n        ))\n        predator_output = predator.forward(response_format={\n            "thinking": "Your step by step reasoning.",\n            "answer": "A single letter, A, B, C, or D."\n        })\n        meeting.chats.append(self.Chat(\n            agent=predator,\n            content=predator_output["thinking"]\n        ))\n        predator_outputs.append(predator_output)  # Collect outputs\n    \n    # Prey agents evaluate and refine predator outputs\n    refined_outputs = []\n    for prey in prey_agents:\n        meeting.chats.append(self.Chat(\n            agent=system,\n            content="Prey agents, please evaluate the predator outputs and suggest refinements."\n        ))\n        for predator_output in predator_outputs:\n            prey_output = prey.forward(response_format={\n                "evaluations": "Your evaluations and suggested refinements to the predator output.",\n                "predator_answer": predator_output["answer"]\n            })\n            meeting.chats.append(self.Chat(\n                agent=prey,\n                content=prey_output["evaluations"]\n            ))\n            refined_outputs.append(prey_output)  # Collect refined outputs\n    \n    # Derive a final answer based on refined outputs\n    final_answers = [output["answer"] for output in refined_outputs if "answer" in output]\n    if not final_answers:\n        return "No valid answer could be determined."\n    # Implement majority voting or selection logic for final answer\n    from collections import Counter\n    final_answer = Counter(final_answers).most_common(1)[0][0]\n    return final_answer  # Return the final answer based on the evaluation.

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
