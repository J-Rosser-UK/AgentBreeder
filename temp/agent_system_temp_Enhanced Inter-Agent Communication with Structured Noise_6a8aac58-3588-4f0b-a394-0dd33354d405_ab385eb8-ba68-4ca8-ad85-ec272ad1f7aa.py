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
        principle_agent = self.Agent(agent_name='Principle Agent', temperature=0.8)
        cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        uncertainty_agent = self.Agent(agent_name='Uncertainty Agent', temperature=0.5)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="structured_noise_meeting")
        meeting.agents.extend([system, principle_agent, cot_agent, uncertainty_agent])
        
        # Get principles involved
        meeting.chats.append(self.Chat(
            agent=system,
            content="What are the principles involved in solving this task? Please list and explain them step by step."
        ))
        
        principle_output = principle_agent.forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        
        meeting.chats.append(self.Chat(
            agent=principle_agent,
            content=principle_output["thinking"] + principle_output["principles"]
        ))
        
        # Solve using the principles
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the principles above, think step by step and solve the task: {task}"
        ))
        
        cot_output = cot_agent.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        
        meeting.chats.append(self.Chat(
            agent=cot_agent,
            content=cot_output["thinking"] + cot_output["answer"]
        ))
        
        # Introduce structured noise to the answer
        noise_output = uncertainty_agent.forward(response_format={
            "plausible_answers": "Provide a range of plausible answers based on a noise model."
        })
        
        # Final decision based on the uncertainty agent's output
        final_answer = noise_output["plausible_answers"]
        return final_answer
    

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
