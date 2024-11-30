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
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        routing_agent = self.Agent(agent_name='Routing Agent', temperature=0.8)
        principle_agent = self.Agent(agent_name='Principle Agent', temperature=0.8)
        cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        shared_memory = []  # Structured shared memory for insights
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="iterative_collaborative_meeting")
        meeting.agents.extend([system, routing_agent, principle_agent, cot_agent])
        
        # Route the task and assign roles based on complexity
        meeting.chats.append(self.Chat(
            agent=system,
            content="Based on the complexity of the task, please assign roles for each agent."
        ))
        routing_output = routing_agent.forward(response_format={
            "assigned_roles": "List of assigned roles for each agent based on task complexity"
        })
        assigned_roles = routing_output["assigned_roles"]
        
        # Get principles involved
        meeting.chats.append(self.Chat(
            agent=system,
            content="What are the principles involved in solving this task? Please list and explain them."
        ))
        principle_output = principle_agent.forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        shared_memory.append(principle_output["thinking"] + principle_output["principles"])  # Store insights
        
        # Allow agents to critique shared memory insights
        for insight in shared_memory:
            meeting.chats.append(self.Chat(
                agent=cot_agent,
                content=f"Please review the following insight: {insight} and provide your thoughts."
            ))
            critique_output = cot_agent.forward(response_format={
                "critique": "Your critique of the insight."
            })
            # Store critiques directly without manual processing
            shared_memory.append(critique_output["critique"])  
        
        # Now solve using the principles based on assigned roles and shared memory
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the question and the involved principles above, and considering the assigned roles: {assigned_roles}, think step by step and then solve the task: {task}. Use the shared insights: {shared_memory}"
        ))
        final_output = cot_agent.forward(response_format={
            "thinking": "Your step by step thinking.",
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
