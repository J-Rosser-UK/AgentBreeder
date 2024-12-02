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
    from functools import wraps
    
    # Custom exception for timeout
    class TimeoutException(Exception):
        pass
    
    # Timeout decorator
    
    def timeout(seconds):
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                if elapsed > seconds:
                    raise TimeoutException("Response exceeded time limit.")
                return result
            return wrapper
        return decorator
    
    class TimeConstrainedAgent:
        def forward(self, task: str) -> str:
            # Create agents
            system = self.Agent(agent_name='system', temperature=0.8)
            principle_agent = self.Agent(agent_name='Principle Agent', temperature=0.8)
            cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
            
            # Setup meeting
            meeting = self.Meeting(meeting_name="time_constrained_meeting")
            meeting.agents.extend([system, principle_agent, cot_agent])
            
            # First get the principles involved
            meeting.chats.append(self.Chat(
                agent=system,
                content="What are the principles involved in solving this task? First think step by step and then list all involved principles and explain them."
            ))
            
            @timeout(10)
            def get_principle_output():
                return principle_agent.forward(response_format={
                    "thinking": "Your step by step thinking about the principles.",
                    "principles": "List and explanation of the principles involved."
                })
            
            try:
                principle_output = get_principle_output()
            except TimeoutException:
                principle_output = {"thinking": "Timeout", "principles": "No principles explained due to timeout."}
            meeting.chats.append(self.Chat(
                agent=principle_agent,
                content=principle_output["thinking"] + principle_output["principles"]
            ))
            
            # Now solve using the principles
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Given the question and the involved principles above, think step by step and then solve the task: {task}"
            ))
            
            @timeout(10)
            def get_final_output():
                return cot_agent.forward(response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                })
            
            try:
                final_output = get_final_output()
            except TimeoutException:
                final_output = {"thinking": "Timeout", "answer": "No answer provided due to timeout."}
            return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
