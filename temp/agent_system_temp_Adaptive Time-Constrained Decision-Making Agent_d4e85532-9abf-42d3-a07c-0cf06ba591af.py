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

    import time
    
    def forward(self, task: str, time_limit: int = 5) -> str:
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="adaptive_time_constrained_meeting")
        meeting.agents.extend([system, cot_agent, final_decision_agent])
        
        # Initial attempt
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please think step by step and then solve the task: {task}"
        ))
        
        start_time = time.time()
        output = cot_agent.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        elapsed_time = time.time() - start_time
        
        # Check if time limit exceeded
        if elapsed_time > time_limit:
            return "Time limit exceeded, providing best attempt based on reasoning so far."
        
        meeting.chats.append(self.Chat(
            agent=cot_agent,
            content=output["thinking"] + output["answer"]
        ))
        
        # Generate diverse solutions
        N_max = 3  # Maximum number of attempts
        for i in range(N_max):
            remaining_time = time_limit - (time.time() - start_time)
            if remaining_time <= 0:
                return "Time limit exceeded during attempts."
            
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Given previous attempts, try to come up with another interesting way to solve the task: {task}. Remaining time: {remaining_time:.2f} seconds"
            ))
            
            output = cot_agent.forward(response_format={
                "thinking": "Your step by step thinking with a new approach.",
                "answer": "A single letter, A, B, C or D."
            })
            
            meeting.chats.append(self.Chat(
                agent=cot_agent,
                content=output["thinking"] + output["answer"]
            ))
        
        # Make final decision
        meeting.chats.append(self.Chat(
            agent=system,
            content="Given all the above solutions, reason over them carefully and provide a final answer."
        ))
        
        final_output = final_decision_agent.forward(response_format={
            "thinking": "Your step by step thinking comparing all solutions.",
            "answer": "A single letter, A, B, C or D."
        })
        
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
