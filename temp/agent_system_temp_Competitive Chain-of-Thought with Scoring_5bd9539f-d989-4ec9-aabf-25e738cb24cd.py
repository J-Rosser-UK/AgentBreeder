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
        # Create a system agent to provide instructions
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create multiple Chain-of-Thought agents
        num_agents = 3
        cot_agents = [
            self.Agent(
                agent_name=f'Chain-of-Thought Agent {i}',
                temperature=0.7
            ) for i in range(num_agents)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="competitive_cot")
        meeting.agents.extend([system] + cot_agents)
        
        # Shared resource: number of attempts
        shared_attempts = 2
        best_answer = None
        best_score = -1  # Initialize best score
        
        for attempt in range(shared_attempts):
            for agent in cot_agents:
                meeting.chats.append(
                    self.Chat(
                        agent=system,
                        content=f"Attempt {attempt + 1}: Please think step by step and solve the task: {task}"
                    )
                )
                output = agent.forward(
                    response_format={
                        "thinking": "Your step by step thinking.",
                        "answer": "A single letter, A, B, C or D."
                    }
                )
                meeting.chats.append(
                    self.Chat(
                        agent=agent,
                        content=output["thinking"]
                    )
                )
                
                # Evaluate the reasoning quality (dummy scoring mechanism)
                score = evaluate_reasoning_quality(output["thinking"])
                if score > best_score:
                    best_answer = output["answer"]
                    best_score = score
        
        return best_answer
    
    
    def evaluate_reasoning_quality(thinking):
        # Implement a scoring mechanism based on criteria such as clarity, relevance, etc.
        # For now, we will use a dummy score based on length as a placeholder.
        return len(thinking)  # Replace with a real scoring mechanism.

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
