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
        trust_agent = self.Agent(agent_name='Trust Evaluation Agent', temperature=0.7)
        
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="role_assignment_meeting")
        meeting.agents.extend([system, routing_agent, trust_agent] + list(expert_agents.values()))
        
        # Route the task
        meeting.chats.append(self.Chat(
            agent=system,
            content="Given the task, please choose an Expert to answer the question. Choose from: Physics, Chemistry, Biology Expert, or Science Generalist."
        ))
        
        routing_output = routing_agent.forward(response_format={
            "choice": "One of: physics, chemistry, biology, or general"
        })
        
        # Select expert based on routing decision
        expert_choice = routing_output["choice"].lower()
        if expert_choice not in expert_agents:
            expert_choice = 'general'
            
        selected_expert = expert_agents[expert_choice]
        
        # Get answers and trust scores from all experts
        answers = []
        trust_scores = []
        for expert in expert_agents.values():
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Please think step by step and then solve the task: {task}"
            ))
            expert_output = expert.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D.",
                "trust_score": "Your trust score between 0 and 1"
            })
            answers.append(expert_output["answer"])
            trust_scores.append(expert_output["trust_score"])
        
        # Evaluate trust scores and decide final answer using weighted voting
        final_answer = trust_agent.forward(response_format={
            "answers": answers,
            "trust_scores": trust_scores
        })
        
        # Ensure final_answer is present in the response
        if "final_answer" not in final_answer:
            raise ValueError("Trust agent did not return a final answer.")
        
        return final_answer["final_answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
