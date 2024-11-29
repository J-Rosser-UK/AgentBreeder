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
        fairness_routing_agent = self.Agent(agent_name='Fairness Routing Agent', temperature=0.8)
        
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="fairness_role_assignment_meeting")
        meeting.agents.extend([system, fairness_routing_agent] + list(expert_agents.values()))
        
        # Evaluate potential experts based on fairness criteria
        meeting.chats.append(self.Chat(
            agent=system,
            content="Given the task, evaluate which Expert would answer the question most fairly. Choose from: Physics, Chemistry, Biology Expert, or Science Generalist."
        ))
        
        routing_output = fairness_routing_agent.forward(response_format={
            "choice": "One of: physics, chemistry, biology, or general"
        })
        
        # Select expert based on routing decision
        expert_choice = routing_output["choice"].lower()
        if expert_choice not in expert_agents:
            expert_choice = 'general'
            
        selected_expert = expert_agents[expert_choice]
        
        # Get answer from selected expert
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please think step by step and then solve the task: {task}"
        ))
        
        expert_output = selected_expert.forward(response_format={
            "thinking": "Your step by step thinking considering fairness.",
            "answer": "A single letter, A, B, C, or D."
        })
        
        # Return the final answer directly from the expert output
        return expert_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
