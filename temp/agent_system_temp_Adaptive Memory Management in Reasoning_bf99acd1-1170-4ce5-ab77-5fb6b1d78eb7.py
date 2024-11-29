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
        # Create agents with adaptive memory management
        system = self.Agent(agent_name='system', temperature=0.8)
        principle_agent = self.Agent(agent_name='Principle Agent', temperature=0.8)
        cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        noise_agent = self.Agent(agent_name='Noise Assessment Agent', temperature=0.5)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="adaptive_memory_meeting")
        meeting.agents.extend([system, principle_agent, cot_agent, noise_agent])
        
        # Dynamic memory assessment
        meeting.chats.append(self.Chat(
            agent=system,
            content="Evaluate the complexity of the task and decide how much memory to allocate for reasoning."
        ))
        
        memory_output = system.forward(response_format={
            "memory_allocation": "How much memory to allocate for this task."
        })
        
        # First get the principles involved
        meeting.chats.append(self.Chat(
            agent=system,
            content="What are the principles involved in solving this task? Please think step by step and explain them, considering your adaptive memory allocation."
        ))
        
        principle_output = principle_agent.forward(response_format={
            "thinking": "Your step by step thinking about the principles, limited to key points based on memory allocation.",
            "principles": "List and explanation of the principles involved, focusing on the most relevant ones."
        })
        
        if not principle_output:
            return "Error: Principle agent did not provide a valid response."
        
        meeting.chats.append(self.Chat(
            agent=principle_agent,
            content=principle_output["thinking"] + principle_output["principles"]
        ))
        
        # Allow agents to communicate their insights after principles are established
        meeting.chats.append(self.Chat(
            agent=cot_agent,
            content="Based on the principles discussed, what are your thoughts on the task? Please keep your insights concise due to memory constraints."
        ))
        
        cot_output = cot_agent.forward(response_format={
            "thinking": "Your reasoning based on the principles, limited to key insights.",
            "insights": "Your insights on the task, focusing on essential points."
        })
        
        if not cot_output:
            return "Error: Chain-of-Thought agent did not provide a valid response."
        
        meeting.chats.append(self.Chat(
            agent=cot_agent,
            content=cot_output["thinking"] + cot_output["insights"]
        ))
        
        # Introduce noise in the reasoning process
        meeting.chats.append(self.Chat(
            agent=noise_agent,
            content="Evaluate how noise might affect the reasoning and insights provided, considering your adaptive memory."
        ))
        
        noise_output = noise_agent.forward(response_format={
            "noise_effect": "Describe how noise impacts the reasoning of the agents, focusing on key aspects."
        })
        
        if not noise_output:
            return "Error: Noise assessment agent did not provide a valid response."
        
        meeting.chats.append(self.Chat(
            agent=noise_agent,
            content=noise_output["noise_effect"]
        ))
        
        # Solve the task considering the principles and noise
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the principles and insights above, think step by step and solve the task: {task}, keeping your response concise and within allocated memory."
        ))
        
        final_output = cot_agent.forward(response_format={
            "thinking": "Your step by step reasoning considering noise and adaptive memory.",
            "answer": "A single letter, A, B, C, or D."
        })
        
        if not final_output:
            return "Error: Final output could not be generated."
        
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
