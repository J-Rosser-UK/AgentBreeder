import random
import pandas

from base import Agent, Meeting, Chat, Wrapper

from sqlalchemy.orm import Session

class AgentSystem:
    def __init__(self, session: Session):
        self.Agent = Wrapper(Agent, session)
        self.Meeting = Wrapper(Meeting, session)
        self.Chat = Wrapper(Chat, session)
        self.session = session

    import numpy as np
    
    def forward(self, task: str) -> str:
        # Create a system agent to provide instructions
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create multiple Chain-of-Thought agents
        N = 3  # Number of agents
        cot_agents = [
            self.Agent(
                agent_name=f'Chain-of-Thought Agent {i}',
                temperature=0.7
            ) for i in range(N)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_trust_evaluation")
        meeting.agents.extend([system] + cot_agents)
        
        # Add system instruction
        meeting.chats.append(
            self.Chat(
                agent=system, 
                content=f"Please think step by step and then solve the task: {task}"
            )
        )
        
        trust_scores = [1.0] * N  # Initialize trust scores to 1.0 for all agents
        outputs = []  # To store outputs from each agent
        
        # Get responses from all agents
        for agent in cot_agents:
            output = agent.forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            outputs.append(output)
            meeting.chats.append(
                self.Chat(
                    agent=agent,
                    content=output["thinking"]
                )
            )
            
            # Update trust scores based on correctness (assume correctness is evaluated externally)
            trust_scores[cot_agents.index(agent)] += 0.1  # Placeholder for correctness evaluation
        
        # Determine the most trusted agent
        trusted_index = np.argmax(trust_scores)
        trusted_output = outputs[trusted_index]
        
        return trusted_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
