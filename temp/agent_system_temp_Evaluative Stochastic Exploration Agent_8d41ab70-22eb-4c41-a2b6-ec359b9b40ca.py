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
        principle_agent = self.Agent(agent_name='Principle Agent', temperature=0.7)
        cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        noise_agent = self.Agent(agent_name='Noise Evaluation Agent', temperature=0.7)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="evaluative_stochastic_exploration_meeting")
        meeting.agents.extend([system, principle_agent, cot_agent, noise_agent])
        
        # Preliminary evaluation: agents provide their reasoning approach
        suitability_outputs = []
        for agent in meeting.agents:
            meeting.chats.append(self.Chat(
                agent=agent,
                content=f"How would you approach the task: {task}? Please provide a brief reasoning."
            ))
            suitability_output = agent.forward(response_format={
                "reasoning": "Your reasoning for the task."
            })
            suitability_outputs.append((agent, suitability_output["reasoning"]))
        
        # Weighted random selection based on suitability
        weights = [1/(i + 1) for i in range(len(suitability_outputs))]  # Example weights
        selected_agents = random.choices(suitability_outputs, weights=weights, k=2)  # Select 2 agents
        
        # Aggregate insights from selected agents
        aggregated_insights = ""
        for selected_agent, _ in selected_agents:
            selected_output = selected_agent.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "insights": "Your insights on the task."
            })
            aggregated_insights += selected_output["thinking"] + selected_output["insights"] + "\n"
        
        # Final evaluation by the system agent based on the aggregated insights
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Based on the aggregated insights provided, think step by step and solve the task: {task}"
        ))
        
        final_output = system.forward(response_format={
            "thinking": "Your final reasoning considering all insights.",
            "answer": "A single letter, A, B, C, or D."
        })
        
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
