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

    def evaluate_strategy(output):
        # Placeholder for evaluation logic
        # This function should return a score based on the reasoning and correctness
        # For now, we will return a dummy score based on the answer correctness
        correct_answer = 'C'  # This should be defined based on the task
        return 1 if output['answer'] == correct_answer else 0  # Simple scoring based on correctness
    
    def forward(self, task: str) -> str:
        # Create agents for the first generation
        system = self.Agent(agent_name='system', temperature=0.8)
        N_agents = 5  # Number of agents in each generation
        generations = 3  # Number of generations
        all_strategies = []  # To store strategies from all generations
    
        for gen in range(generations):
            meeting = self.Meeting(meeting_name=f'generation_{gen + 1}')
            agents = [self.Agent(agent_name=f'Agent_{i + 1}_Gen_{gen + 1}', temperature=0.8) for i in range(N_agents)]
            meeting.agents.extend([system] + agents)
    
            # Each agent attempts to solve the task
            for agent in agents:
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f'Agent {agent.agent_name}, please solve the task step by step: {task}'
                ))
                output = agent.forward(response_format={
                    'thinking': 'Your step by step reasoning.',
                    'answer': 'A, B, C, or D.'
                })
                meeting.chats.append(self.Chat(
                    agent=agent,
                    content=output['thinking'] + output['answer']
                ))
                # Store the successful strategies with a score
                score = evaluate_strategy(output)
                all_strategies.append((output, score))
    
            # Select top strategies for the next generation
            all_strategies.sort(key=lambda x: x[1], reverse=True)  # Sort by score
            top_strategies = all_strategies[:N_agents]  # Keep only top N strategies
    
            # Critique and share strategies with the next generation
            if gen < generations - 1:  # No need to critique after the last generation
                for agent in agents:
                    meeting.chats.append(self.Chat(
                        agent=system,
                        content=f'Agent {agent.agent_name}, critique your performance and suggest improvements.'
                    ))
                    critique_output = agent.forward(response_format={
                        'critique': 'Your critique of your previous answer.'
                    })
                    meeting.chats.append(self.Chat(
                        agent=agent,
                        content=critique_output['critique']
                    ))
    
        # Final decision from the last generation
        meeting.chats.append(self.Chat(
            agent=system,
            content='Review all strategies from previous generations and provide a final answer.'
        ))
        final_output = agents[-1].forward(response_format={
            'thinking': 'Your reasoning based on all strategies from previous generations.',
            'answer': 'A, B, C, or D.'
        })
        return final_output['answer']

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
