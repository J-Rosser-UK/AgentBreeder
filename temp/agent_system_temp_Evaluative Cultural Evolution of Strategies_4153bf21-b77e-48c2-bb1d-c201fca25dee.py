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
        # Create agents for the first generation
        system = self.Agent(agent_name='system', temperature=0.8)
        cot_agents = [self.Agent(agent_name=f'Chain-of-Thought Agent {i}', temperature=0.8) for i in range(3)]
        
        # Setup meeting for the first generation
        meeting = self.Meeting(meeting_name="evaluative_cultural_evolution_meeting")
        meeting.agents.extend([system] + cot_agents)
        
        generations = 2  # Number of generations
        successful_strategies = []  # To store successful strategies from each generation
        effectiveness_scores = []  # To store effectiveness scores for strategies
        
        for gen in range(generations):
            # Initial task solving
            for agent in cot_agents:
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f"Please think step by step and solve the task: {task}"
                ))
                output = agent.forward(response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C, or D.",
                    "effectiveness": "Score of effectiveness"
                })
                meeting.chats.append(self.Chat(
                    agent=agent,
                    content=output["thinking"] + output["answer"]
                ))
                successful_strategies.append(output["thinking"])  # Collecting successful strategies
                effectiveness_scores.append(output["effectiveness"])  # Collecting effectiveness scores
            
            # Share strategies among agents
            shared_strategy = "; ".join(successful_strategies)  # Combine strategies for sharing
            for agent in cot_agents:
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f"Based on the previous generation's strategies: {shared_strategy}, consider these insights in your next attempt."
                ))
            
            # Create new agents for the next generation based on shared strategies
            prioritized_agents = sorted(zip(cot_agents, effectiveness_scores), key=lambda x: x[1], reverse=True)[:3]  # Select top strategies
            cot_agents = [self.Agent(agent_name=f'Chain-of-Thought Agent {i + 3}', temperature=0.8) for i in range(len(prioritized_agents))]  # New generation
            meeting.agents.extend(cot_agents)
            successful_strategies.clear()  # Clear strategies for the next generation
            effectiveness_scores.clear()  # Clear scores for the next generation
        
        # Final decision based on insights from the last generation
        final_answers = []
        for agent in cot_agents:
            output = agent.forward(response_format={
                "thinking": "Your final step by step thinking.",
                "answer": "A single letter, A, B, C, or D."
            })
            final_answers.append(output["answer"])  
        
        # Aggregate answers by majority voting
        final_answer = max(set(final_answers), key=final_answers.count)
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
