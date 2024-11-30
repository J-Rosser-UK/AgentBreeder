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
        cot_agents = [
            self.Agent(agent_name=f'Chain-of-Thought Agent {i}', temperature=0.8) for i in range(3)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name='negotiation_meeting')
        meeting.agents.extend([system] + cot_agents)
        
        # Initial instruction to agents
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please assess the task: {task} and discuss your reasoning with each other."
        ))
        
        # Each agent presents their reasoning and confidence levels
        agent_outputs = []
        for agent in cot_agents:
            output = agent.forward(response_format={
                "reasoning": "Your reasoning for the task.",
                "confidence": "Your confidence level (0-1)"
            })
            # Validate and convert confidence level
            try:
                confidence = float(output["confidence"])
                agent_outputs.append((output["reasoning"], confidence))
            except (ValueError, TypeError):
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f"{agent.agent_name} provided an invalid confidence level. Skipping this agent."
                ))
        
        # Negotiation phase
        consensus_reached = False
        while not consensus_reached:
            # Evaluate confidence levels
            high_confidence_agents = [output for output in agent_outputs if output[1] >= 0.5]
            if len(high_confidence_agents) > 1:
                # If multiple agents have high confidence, they can form a coalition
                consensus_reached = True
                final_answer = high_confidence_agents[0][0]  # Take the reasoning of the first high confidence agent
            else:
                # If no consensus, allow agents to discuss further
                meeting.chats.append(self.Chat(
                    agent=system,
                    content="No consensus reached. Please discuss your reasoning further to reach an agreement."
                ))
                # Re-evaluate outputs after discussion
                for agent in cot_agents:
                    output = agent.forward(response_format={
                        "reasoning": "Reassess your reasoning after discussion.",
                        "confidence": "Your new confidence level (0-1)"
                    })
                    try:
                        confidence = float(output["confidence"])
                        agent_outputs.append((output["reasoning"], confidence))
                    except (ValueError, TypeError):
                        meeting.chats.append(self.Chat(
                            agent=system,
                            content=f"{agent.agent_name} provided an invalid confidence level. Skipping this agent."
                        ))
    
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
