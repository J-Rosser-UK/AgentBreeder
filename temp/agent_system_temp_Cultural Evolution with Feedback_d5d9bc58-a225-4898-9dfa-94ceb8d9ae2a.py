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
        # Create agents with defined roles
        base_agent = self.Agent(agent_name='Base Agent', temperature=0.7)
        successor_agent = self.Agent(agent_name='Successor Agent', temperature=0.7)
        
        # Setup meeting for knowledge sharing
        meeting = self.Meeting(meeting_name="cultural_evolution_meeting")
        meeting.agents.extend([base_agent, successor_agent])
        
        # Base agent attempts to solve the task
        meeting.chats.append(self.Chat(
            agent=base_agent,
            content=f"Please solve this task: {task}"
        ))
        
        base_output = base_agent.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C, or D.",
            "confidence": "Your confidence level (0-1)."
        })
        
        # Record the base agent's output
        meeting.chats.append(self.Chat(
            agent=base_agent,
            content=base_output["thinking"] + base_output["answer"]
        ))
        
        # Successor agent reviews the base agent's output and reflects
        meeting.chats.append(self.Chat(
            agent=successor_agent,
            content="Reflecting on the previous agent's output, do you have any clarifying questions?"
        ))
        
        # Successor agent asks clarifying questions
        clarifying_output = successor_agent.forward(response_format={
            "questions": "Any questions for clarification?"
        })
        
        # If questions are asked, respond to them
        if clarifying_output["questions"]:
            meeting.chats.append(self.Chat(
                agent=base_agent,
                content=f"Here are the answers to your questions: {clarifying_output["questions"]}"
            ))
        
        # Successor agent attempts to solve the task again
        successor_output = successor_agent.forward(response_format={
            "thinking": "Your step by step thinking incorporating learned strategies.",
            "answer": "A single letter, A, B, C, or D.",
            "confidence": "Your confidence level (0-1)."
        })
        
        # Aggregate the answers by selecting the one with the highest confidence, handling ties
        if base_output["confidence"] > successor_output["confidence"]:
            final_answer = base_output["answer"]
        elif successor_output["confidence"] > base_output["confidence"]:
            final_answer = successor_output["answer"]
        else:
            final_answer = base_output["answer"]  # or choose randomly if preferred
        
        return final_answer
    

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
