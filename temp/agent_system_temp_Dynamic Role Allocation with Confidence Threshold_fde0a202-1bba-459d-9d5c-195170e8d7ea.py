import random
import numpy as np
import pandas

from mocked_base import MockAgent as Agent

from mocked_base import MockMeeting as Meeting

from mocked_base import MockChat as Chat

from sqlalchemy.orm import Session

class AgentSystem:
    def __init__(self, session: Session = None):
        self.Agent = Agent
        self.Meeting = Meeting
        self.Chat = Chat
        self.session = session

    def forward(self, task: str) -> str:
        # Create a system agent to provide instructions
        system = self.Agent(agent_name='system', temperature=0.8)
        routing_agent = self.Agent(agent_name='Routing Agent', temperature=0.8)
        
        # Create candidate agents with different specialties
        candidate_agents = [
            self.Agent(agent_name='Expert Agent 1', temperature=0.7),
            self.Agent(agent_name='Expert Agent 2', temperature=0.7),
            self.Agent(agent_name='Expert Agent 3', temperature=0.7)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_role_allocation")
        meeting.agents.extend([system, routing_agent] + candidate_agents)
        
        # Gather confidence levels from candidate agents
        confidence_levels = []
        for agent in candidate_agents:
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"How confident are you in solving the task: {task}? Please respond with a confidence level from 0 to 1."
            ))
            output = agent.forward(response_format={"confidence": "A number from 0 to 1"})
            confidence_levels.append((agent, output["confidence"]))
    
        # Set a confidence threshold
        confidence_threshold = 0.5
        filtered_agents = [(agent, confidence) for agent, confidence in confidence_levels if confidence >= confidence_threshold]
    
        # If no agent meets the confidence threshold, return a message
        if not filtered_agents:
            return "No agent is confident enough to solve the task."
    
        # Select the agent with the highest confidence, with tie-breaking logic
        best_agent = max(filtered_agents, key=lambda x: (x[1], x[0].agent_id))[0]
        
        # Task instruction to the selected agent
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please think step by step and then solve the task: {task}"
        ))
        
        # Get response from the best agent
        output = best_agent.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        
        # Record the agent's response in the meeting
        meeting.chats.append(
            self.Chat(
                agent=best_agent,
                content=output["thinking"]
            )
        )
        
        return output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
