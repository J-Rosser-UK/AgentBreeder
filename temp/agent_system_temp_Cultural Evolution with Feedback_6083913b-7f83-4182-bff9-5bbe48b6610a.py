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

    from sqlalchemy.ext.declarative import declarative_base
    
    # Assuming CustomBase is a SQLAlchemy declarative base
    CustomBase = declarative_base()
    
    class Agent(CustomBase):
        __tablename__ = 'agent'
    
        agent_id = CustomColumn(String, primary_key=True, default=lambda: str(uuid.uuid4()), label="The agent's unique identifier (UUID).")
        agent_name = CustomColumn(String, label="The agent's name.")
        agent_backstory = CustomColumn(String, label="A long description of the agent's backstory.")
        model = CustomColumn(String, label="The LLM model to be used.")
        temperature = CustomColumn(Float, default=0.7, label="The sampling temperature. The higher the temperature, the more creative the responses.")
        agent_timestamp = CustomColumn(DateTime, default=datetime.datetime.now(), label="The timestamp of the agent's creation.")
        successful_strategies = CustomColumn(JSON, default=list, label="List of successful strategies.")  # Added attribute
    
        def __init__(self, session, agent_name, model='gpt-4o-mini', temperature=0.5):
            super().__init__(session, agent_name=agent_name, model=model, temperature=temperature)
            self.successful_strategies = []  # Initialize successful strategies
    
        # Existing methods...
    
    def forward(self, task: str) -> str:
        # Function to collect successful strategies from previous generations
        def collect_successful_strategies(previous_agents):
            strategies = []
            for agent in previous_agents:
                strategies.extend(agent.successful_strategies)
            return strategies
    
        # Function to get previous generation agents
        def get_previous_agents():
            # This function should return a list of agents from the previous generation
            return []  # Placeholder for actual implementation
    
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        evolution_agent = self.Agent(agent_name='Evolution Agent', temperature=0.7)
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="cultural_evolution_meeting")
        meeting.agents.extend([system, evolution_agent] + list(expert_agents.values()))
        
        # Collect strategies from previous generation
        previous_agents = get_previous_agents()  # Populate from previous generation
        successful_strategies = collect_successful_strategies(previous_agents)
        
        # Share strategies with current agents and evaluate them
        for expert in expert_agents.values():
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"{expert.agent_name}, consider the following strategies: {successful_strategies}. Now think step by step and solve the task: {task}"
            ))
            expert_output = expert.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C or D."
            })
            meeting.chats.append(self.Chat(
                agent=expert,
                content=expert_output["thinking"]
            ))
            # Record successful strategies based on performance
            if expert_output['answer'] in ['A', 'B', 'C', 'D']:
                expert.successful_strategies.append(expert_output['thinking'])
    
        # Evaluate the answers and select the best one
        evaluation_results = []
        for expert in expert_agents.values():
            evaluation = expert.evaluate_strategies(successful_strategies)
            evaluation_results.append(evaluation)
    
        # Filter successful strategies based on evaluations
        filtered_strategies = [s for s in successful_strategies if s in evaluation_results]  # Placeholder logic
    
        # Select the best strategy based on evaluations
        final_answer = expert_agents['general'].forward(response_format={
            "final_answer": "Provide the best answer based on the experts' reasoning."
        })
        return final_answer['final_answer']  # Return the final answer based on the evaluation.

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
