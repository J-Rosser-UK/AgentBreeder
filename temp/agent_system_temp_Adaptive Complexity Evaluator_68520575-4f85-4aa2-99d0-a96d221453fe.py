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

    import json
    
    def forward(self, task: str) -> str:
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        primary_agent = self.Agent(agent_name='Primary Decision Agent', temperature=0.8)
        expert_agents = {
            'simple': self.Agent(agent_name='Simple Task Expert', temperature=0.8),
            'moderate': self.Agent(agent_name='Moderate Task Expert', temperature=0.8),
            'complex': self.Agent(agent_name='Complex Task Expert', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name='adaptive_complexity_meeting')
        meeting.agents.extend([system, primary_agent] + list(expert_agents.values()))
        
        # Primary agent evaluates task complexity
        meeting.chats.append(self.Chat(
            agent=system,
            content="Please evaluate the complexity of the task based on predefined criteria and categorize it as simple, moderate, or complex."
        ))
        
        complexity_output = primary_agent.forward(response_format={
            "complexity": "One of: simple, moderate, complex"
        })
        
        expert_choice = complexity_output["complexity"].lower()
        if expert_choice not in expert_agents:
            return json.dumps({"error": "Invalid complexity category. No expert selected."})
        
        selected_expert = expert_agents[expert_choice]
        
        # Get answer from selected expert
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please think step by step and then solve the task: {task}"
        ))
        
        expert_output = selected_expert.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C, or D."
        })
        
        return json.dumps({"answer": expert_output["answer"]})

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
