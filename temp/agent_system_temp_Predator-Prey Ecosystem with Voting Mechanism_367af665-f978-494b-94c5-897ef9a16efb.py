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
        predator_agents = [
            self.Agent(agent_name='Predator Expert 1', temperature=0.8),
            self.Agent(agent_name='Predator Expert 2', temperature=0.8)
        ]
        prey_agents = [
            self.Agent(agent_name='Prey Expert 1', temperature=0.7),
            self.Agent(agent_name='Prey Expert 2', temperature=0.7)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="predator_prey_meeting")
        meeting.agents.extend([system] + predator_agents + prey_agents)
        
        # Round-robin interaction: Predators challenge each prey
        for prey in prey_agents:
            for predator in predator_agents:
                meeting.chats.append(self.Chat(
                    agent=predator,
                    content=f"{predator.agent_name}, please provide your best reasoning for the task: {task}"
                ))
                
                prey_output = prey.forward(response_format={
                    "thinking": "Your step by step reasoning.",
                    "answer": "A single letter, A, B, C, or D."
                })
                meeting.chats.append(self.Chat(
                    agent=prey,
                    content=prey_output["thinking"]
                ))
                
                # Predators evaluate prey responses
                meeting.chats.append(self.Chat(
                    agent=predator,
                    content=f"Evaluate the response from {prey.agent_name}: {prey_output[\"answer\"]}"
                ))
        
        # Collect evaluations from predators
        predator_evaluations = []
        for predator in predator_agents:
            evaluation_output = predator.forward(response_format={
                "evaluation": "Your evaluation of the responses."
            })
            predator_evaluations.append(evaluation_output["evaluation"])
        
        # Final evaluation by the system agent
        meeting.chats.append(self.Chat(
            agent=system,
            content="Based on the evaluations, provide a final answer."
        ))
        final_output = system.forward(response_format={
            "final_evaluations": predator_evaluations,
            "final_answer": "A single letter, A, B, C, or D."
        })
        
        return final_output["final_answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
