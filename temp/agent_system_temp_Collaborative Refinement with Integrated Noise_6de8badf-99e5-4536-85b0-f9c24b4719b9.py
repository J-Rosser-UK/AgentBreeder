import random
import pandas

from base import Agent, Meeting, Chat

from sqlalchemy import Session

class AgentSystem:
    def __init__(self, session: Session):
        self.Agent = Wrapper(Agent, session)
        self.Meeting = Wrapper(Meeting, session)
        self.Chat = Wrapper(Chat, session)
        self.session = session

    def forward(self, task: str) -> str:
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        collaborator1 = self.Agent(agent_name='Collaborator 1', temperature=0.7)
        collaborator2 = self.Agent(agent_name='Collaborator 2', temperature=0.7)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaboration_with_noise")
        meeting.agents.extend([system, collaborator1, collaborator2])
        
        # Initial instruction from system
        meeting.chats.append(self.Chat(
            agent=system,
            content="Please think step by step about the task: {task} and share your thoughts with each other."
        ))
        
        # Collaborators think and share their reasoning
        output1 = collaborator1.forward(response_format={
            "thinking": "Your step by step reasoning with integrated noise.",
            "answer": "A single letter, A, B, C or D."
        })
        
        output2 = collaborator2.forward(response_format={
            "thinking": "Your step by step reasoning with integrated noise.",
            "answer": "A single letter, A, B, C or D."
        })
        
        # Process outputs before final decision
        final_thinking = f"Collaborator 1 thinks: {output1['thinking']}\nCollaborator 2 thinks: {output2['thinking']}\n" 
        final_answer = "A"  # Placeholder for consensus logic, to be replaced with actual logic.
        
        # Final decision based on collaborative reasoning
        meeting.chats.append(self.Chat(
            agent=system,
            content="Based on the reasoning from both collaborators, the final decision is: {final_answer}."
        ))
        
        return final_answer
    

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
