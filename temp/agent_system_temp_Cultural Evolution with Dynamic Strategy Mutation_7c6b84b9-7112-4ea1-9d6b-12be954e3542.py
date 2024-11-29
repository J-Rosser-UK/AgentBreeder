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

    def forward(self, task: str) -> str:\n    N_generations = 5  # Number of generations\n    N_agents = 5  # Number of agents per generation\n    successful_strategies = []  # Store successful strategies\n\n    for generation in range(N_generations):\n        agents = [self.Agent(agent_name=f'Agent Gen{generation} {i}', temperature=0.8) for i in range(N_agents)]\n        meeting = self.Meeting(meeting_name=f"cultural_evolution_gen_{generation}")\n        meeting.agents.extend(agents)\n\n        # Each agent attempts to solve the task\n        for agent in agents:\n            meeting.chats.append(self.Chat(\n                agent=agent,\n                content=f"Please think step by step and then solve the task: {task}"\n            ))\n\n            output = agent.forward(response_format={\n                "thinking": "Your step by step thinking.",\n                "answer": "A single letter, A, B, C, or D."\n            })\n\n            meeting.chats.append(self.Chat(\n                agent=agent,\n                content=output["thinking"]\n            ))\n\n            # Record successful strategies based on reasoning quality\n            if output["answer"] in ['A', 'B', 'C', 'D']:  # Assuming all answers are valid\n                successful_strategies.append((output["thinking"], output["answer"], agent))\n\n        # Create new generation based on successful strategies\n        if successful_strategies:\n            # Select best strategies based on reasoning quality\n            successful_strategies.sort(key=lambda x: len(x[0]), reverse=True)  # Sort by length of reasoning\n            next_generation_strategies = successful_strategies[:len(successful_strategies)//2]\n\n            # Mutate strategies for next generation\n            for i in range(N_agents):\n                strategy = next_generation_strategies[i % len(next_generation_strategies)]\n                # Introduce mutation logic here, e.g., changing temperature\n                new_agent = self.Agent(agent_name=f'Agent Gen{generation+1} {i}', temperature=0.7 + 0.1 * (i % 3))  # Slightly varied temperature\n                meeting.agents.append(new_agent)\n                meeting.chats.append(self.Chat(\n                    agent=new_agent,\n                    content=f"Using the reasoning: {strategy[0]}, please think step by step and solve the task: {task}"\n                ))\n\n    # Final decision based on last generation's outputs\n    final_answers = [output["answer"] for _, output, _ in successful_strategies]  # Collect only the answers\n\n    # Use majority voting among final answers\n    from collections import Counter\n    answer_counts = Counter(final_answers)\n    final_answer = answer_counts.most_common(1)[0][0]  # Get the most common answer\n    return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
