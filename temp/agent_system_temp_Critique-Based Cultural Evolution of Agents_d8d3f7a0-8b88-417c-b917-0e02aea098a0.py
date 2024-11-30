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
        # Create initial population of agents
        system = self.Agent(agent_name='system', temperature=0.8)
        N_agents = 5  # Number of agents in the population
        agents = [self.Agent(agent_name=f'Agent {i}', temperature=0.8) for i in range(N_agents)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="critique_cultural_evolution")
        meeting.agents.extend([system] + agents)
        
        generations = 3  # Number of generations
        N_max = 3  # Maximum number of attempts per agent
        
        for generation in range(generations):
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Generation {generation + 1}: Each agent will attempt to solve the task: {task}"
            ))
            
            # Each agent attempts to solve the task
            for agent in agents:
                for attempt in range(N_max):
                    output = agent.forward(response_format={
                        "thinking": "Your step by step thinking.",
                        "answer": "A single letter, A, B, C or D."
                    })
                    meeting.chats.append(self.Chat(
                        agent=agent,
                        content=output["thinking"] + output["answer"]
                    ))
            
            # Evaluate performance and select the best agents
            performance = []
            for agent in agents:
                # Assume evaluate_agent is a function that returns a score based on correctness and reasoning
                score = evaluate_agent(agent)  # This should evaluate based on the correctness of the answer
                performance.append((agent, score))  
            
            # Sort agents by performance
            performance.sort(key=lambda x: x[1], reverse=True)
            top_agents = [x[0] for x in performance[:N_agents // 2]]  # Select top half
            
            # Create new generation
            new_agents = []
            for i in range(N_agents):
                parent = top_agents[i % len(top_agents)]  # Select parents from top agents
                new_agent = self.Agent(agent_name=f'New Agent {i}', temperature=parent.temperature + random.uniform(-0.1, 0.1))  # Create a new agent with slight temperature variation
                new_agents.append(new_agent)
            
            agents = new_agents  # Replace old agents with new generation
        
        # Final decision from the best agent
        best_agent = agents[0]  # Assume the best agent is the first one
        final_output = best_agent.forward(response_format={
            "thinking": "Your final reasoning based on all inputs.",
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
