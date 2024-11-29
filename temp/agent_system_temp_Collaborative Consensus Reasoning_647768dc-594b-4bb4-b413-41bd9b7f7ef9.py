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

    def forward(self, task: str) -> str:
        import random
        from collections import Counter
        
        # Create a system agent to provide instructions
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create multiple Chain-of-Thought agents
        num_agents = 3
        cot_agents = [self.Agent(
            agent_name=f'Chain-of-Thought Agent {i}',
            temperature=0.7
        ) for i in range(num_agents)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_consensus")
        meeting.agents.extend([system] + cot_agents)
        
        # Add system instruction
        meeting.chats.append(
            self.Chat(
                agent=system, 
                content=f"Please think step by step and then solve the task: {task}"
            )
        )
        
        # Each agent thinks and shares their reasoning with noise
        responses = []
        for agent in cot_agents:
            output = agent.forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            # Introduce noise in the reasoning process
            noise = random.choice([-0.1, 0, 0.1])  # Simulating uncertainty
            modified_thinking = output["thinking"] + f" (with noise: {noise})"
            responses.append(output["answer"])
            meeting.chats.append(
                self.Chat(
                    agent=agent, 
                    content=modified_thinking
                )
            )
        
        # Aggregate answers using majority voting
        final_answer = Counter(responses).most_common(1)[0][0]  # Get the most common answer
        
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
