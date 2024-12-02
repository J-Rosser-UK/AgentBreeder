import random
import numpy as np
import pandas

from mocked_base import MockAgent as Agent

from mocked_base import MockMeeting as Meeting

from mocked_base import MockChat as Chat

from sqlalchemy.orm import Session

class AgentSystem:
    def __init__(self, session: Session):
        self.Agent = Agent
        self.Meeting = Meeting
        self.Chat = Chat
        self.session = session

    def forward(self, task: str) -> str:
        import random
        
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="adaptive_exploration_meeting")
        meeting.agents.extend([system, cot_agent, final_decision_agent])
        
        N_max = 3  # Maximum number of attempts
        previous_outputs = []  # Store previous outputs for feedback
        
        # Initial attempt
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please think step by step and then solve the task: {task}"
        ))
        
        output = cot_agent.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        
        meeting.chats.append(self.Chat(
            agent=cot_agent,
            content=output["thinking"] + output["answer"]
        ))
        previous_outputs.append(output)  # Store initial output
        
        # Generate diverse solutions with adaptive feedback
        for i in range(N_max):
            # Evaluate previous outputs to decide on next approach
            if i > 0:
                feedback = evaluate_feedback(previous_outputs)
                approach = select_approach(feedback)  # Define this function based on feedback
            else:
                approach = "original"  # Default first approach
    
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Given previous attempts, try to come up with another interesting way to solve the task using the {approach} approach: {task}"
            ))
            
            output = cot_agent.forward(response_format={
                "thinking": "Your step by step thinking with a new approach.",
                "answer": "A single letter, A, B, C, or D."
            })
            
            meeting.chats.append(self.Chat(
                agent=cot_agent,
                content=output["thinking"] + output["answer"]
            ))
            previous_outputs.append(output)  # Store output for feedback
        
        # Make final decision
        meeting.chats.append(self.Chat(
            agent=system,
            content="Given all the above solutions, reason over them carefully and provide a final answer."
        ))
        
        final_output = final_decision_agent.forward(response_format={
            "thinking": "Your step by step thinking comparing all solutions.",
            "answer": "A single letter, A, B, C, or D."
        })
        
        return final_output["answer"]
    

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
