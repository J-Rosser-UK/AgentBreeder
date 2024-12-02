import random
import numpy as np
import pandas

from mocked_base import MockAgent as Agent

from mocked_base import MockMeeting as Meeting

from mocked_base import MockChat as Chat

from sqlalchemy.orm import Session

class AgentSystem:
    def __init__(self, session: Session = None):
        self.Agent = Agent
        self.Meeting = Meeting
        self.Chat = Chat
        self.session = session

    def forward(self, task: str) -> str:
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.7)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name='time_constrained_meeting')
        meeting.agents.extend([system, cot_agent])
        
        # Set a time limit for decision making
        time_limit = 3  # Number of iterations allowed for reasoning
        
        # Initial instruction from the system
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"You have {time_limit} iterations to think step by step and then solve the task: {task}."
        ))
        
        # Reasoning loop under time constraint
        reasoning_steps = []  # List to accumulate reasoning outputs
        for i in range(time_limit):
            output = cot_agent.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            })
            reasoning_steps.append(output["thinking"])  # Store each reasoning step
            meeting.chats.append(self.Chat(
                agent=cot_agent,
                content=output["thinking"]
            ))
            
        # Final output after time constraint
        final_answer = output["answer"]  # Use the last answer as the final answer
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Final reasoning steps:\n{'\n'.join(reasoning_steps)}\nFinal answer: {final_answer}"
        ))
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
