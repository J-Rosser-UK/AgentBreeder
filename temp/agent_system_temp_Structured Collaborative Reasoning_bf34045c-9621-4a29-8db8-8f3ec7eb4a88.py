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

    def forward(self, task: str) -> str:
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        review_agent = self.Agent(agent_name='Collaborative Review Agent', temperature=0.6)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="structured_collaborative_meeting")
        meeting.agents.extend([system, cot_agent, review_agent])
        
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
        
        # Collaborative review phase
        meeting.chats.append(self.Chat(
            agent=system,
            content="Let's review the previous reasoning and answer. What can we improve or refine based on the task?"
        ))
        
        review_output = review_agent.forward(response_format={
            "review_thinking": "Your collaborative review of the reasoning.",
            "suggestion": "Suggested modifications to the answer."
        })
        
        meeting.chats.append(self.Chat(
            agent=review_agent,
            content=review_output["review_thinking"] + " Suggested Change: " + review_output["suggestion"]
        ))
        
        # Final answer derived directly from review suggestions
        return review_output["suggestion"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
