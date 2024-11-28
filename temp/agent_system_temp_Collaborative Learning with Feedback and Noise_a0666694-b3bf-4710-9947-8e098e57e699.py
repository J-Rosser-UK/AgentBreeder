import random
import pandas

from base import Agent, Meeting, Chat

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
        principle_agent = self.Agent(agent_name='Principle Agent', temperature=0.8)
        cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        noise_agent = self.Agent(agent_name='Noise Agent', temperature=0.5)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="feedback_noise_meeting")
        meeting.agents.extend([system, principle_agent, cot_agent, noise_agent])
        
        # First get the principles involved
        meeting.chats.append(self.Chat(
            agent=system,
            content="What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
        ))
        
        principle_output = principle_agent.forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        
        meeting.chats.append(self.Chat(
            agent=principle_agent,
            content=principle_output["thinking"] + principle_output["principles"]
        ))
        
        # Now solve using the principles
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the question and the involved principles above, think step by step and then solve the task: {task}"
        ))
        
        cot_output = cot_agent.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        
        # Introduce noise and provide feedback
        noise_output = noise_agent.forward(response_format={
            "original_answer": cot_output["answer"],
            "feedback": "Evaluate the answer and introduce variability."
        })
        
        # Use feedback to refine CoT agent's output
        refined_output = cot_agent.forward(response_format={
            "thinking": "Given the feedback, refine your answer based on the original output.",
            "refined_answer": noise_output["original_answer"]
        })
        
        return refined_output["refined_answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
