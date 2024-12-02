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

    import time
    
    def forward(self, task: str, time_limit: int = 5) -> str:
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        principle_agent = self.Agent(agent_name='Principle Agent', temperature=0.8)
        cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="real_time_decision_meeting")
        meeting.agents.extend([system, principle_agent, cot_agent])
        
        # First get the principles involved
        meeting.chats.append(self.Chat(
            agent=system,
            content="What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
        ))
        
        # Get principles with time constraint
        start_time = time.time()
        principle_output = principle_agent.forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            principle_output = {"thinking": "Time exceeded", "principles": "Fallback response due to time limit."}
        
        meeting.chats.append(self.Chat(
            agent=principle_agent,
            content=principle_output["thinking"] + principle_output["principles"]
        ))
        
        # Now solve using the principles
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the question and the involved principles above, think step by step and then solve the task: {task}"
        ))
        
        # Get final answer with time constraint
        start_time = time.time()
        final_output = cot_agent.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        elapsed_time = time.time() - start_time
        if elapsed_time > time_limit:
            final_output = {"thinking": "Time exceeded", "answer": "Fallback response due to time limit."}
        
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
