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
        principle_agent = self.Agent(agent_name='Principle Agent', temperature=0.8)
        cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        noise_agent = self.Agent(agent_name='Noise Assessment Agent', temperature=0.5)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_role_allocation_feedback_meeting")
        meeting.agents.extend([system, principle_agent, cot_agent, noise_agent])
        
        # Evaluate agent performance from previous tasks
        performance_metrics = {agent.agent_name: self.evaluate_performance(agent) for agent in meeting.agents}
        
        # Dynamically assign roles based on performance
        assigned_roles = self.assign_roles(performance_metrics)
        
        # First get the principles involved
        meeting.chats.append(self.Chat(
            agent=system,
            content="What are the principles involved in solving this task? Please think step by step and explain them."
        ))
        
        principle_output = assigned_roles['Principle Agent'].forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        
        meeting.chats.append(self.Chat(
            agent=assigned_roles['Principle Agent'],
            content=principle_output["thinking"] + principle_output["principles"]
        ))
        
        # Allow agents to communicate their insights after principles are established
        meeting.chats.append(self.Chat(
            agent=assigned_roles['Chain-of-Thought Agent'],
            content="Based on the principles discussed, what are your thoughts on the task?"
        ))
        
        cot_output = assigned_roles['Chain-of-Thought Agent'].forward(response_format={
            "thinking": "Your reasoning based on the principles.",
            "insights": "Your insights on the task."
        })
        
        meeting.chats.append(self.Chat(
            agent=assigned_roles['Chain-of-Thought Agent'],
            content=cot_output["thinking"] + cot_output["insights"]
        ))
        
        # Introduce noise in the reasoning process
        meeting.chats.append(self.Chat(
            agent=assigned_roles['Noise Assessment Agent'],
            content="Evaluate how noise might affect the reasoning and insights provided."
        ))
        
        noise_output = assigned_roles['Noise Assessment Agent'].forward(response_format={
            "noise_effect": "Describe how noise impacts the reasoning of the agents."
        })
        
        meeting.chats.append(self.Chat(
            agent=assigned_roles['Noise Assessment Agent'],
            content=noise_output["noise_effect"]
        ))
        
        # Solve the task considering the principles and noise
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the principles and insights above, think step by step and solve the task: {task}"
        ))
        
        final_output = assigned_roles['Chain-of-Thought Agent'].forward(response_format={
            "thinking": "Your step by step reasoning considering noise.",
            "answer": "A single letter, A, B, C, or D."
        })
        
        # Feedback loop to adjust roles based on performance
        self.provide_feedback(meeting.agents)
        
        return final_output["answer"]
    
    # Methods for performance evaluation and role assignment
    def evaluate_performance(self, agent):
        # Logic to evaluate agent's past performance
        # Placeholder logic for performance scoring
        # In practice, this should compare the agent's answers to a set of correct answers
        return random.randint(0, 100)  # Example: random score for demonstration
    
    
    def assign_roles(self, performance_metrics):
        # Logic to assign roles based on performance metrics
        return {agent_name: agent for agent_name, agent in performance_metrics.items() if performance_metrics[agent_name] > 50}  # Example: assign roles to agents with scores above 50
    
    
    def provide_feedback(self, agents):
        # Logic for agents to provide feedback on each other's performance
        for agent in agents:
            feedback_output = agent.forward(response_format={
                "feedback": "Your feedback on the performance of other agents."
            })
            # Process feedback if necessary
            # Here we can store or use the feedback to adjust future roles
            pass

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
