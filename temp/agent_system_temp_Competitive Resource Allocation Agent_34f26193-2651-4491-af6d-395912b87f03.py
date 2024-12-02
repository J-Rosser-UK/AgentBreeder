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
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        routing_agent = self.Agent(agent_name='Routing Agent', temperature=0.8)
        
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="role_assignment_meeting")
        meeting.agents.extend([system, routing_agent] + list(expert_agents.values()))
        
        # Initialize shared resource (e.g., tokens)
        shared_resource = {'tokens': 5}  # Limited tokens available
        scores = {expert: 0 for expert in expert_agents}  # Initialize scores for each expert
        request_queue = []  # Queue to manage resource requests
        
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
        
        # Add the expert to the request queue
        request_queue.append(selected_expert)
        
        # Process requests in the queue
        if request_queue:
            current_expert = request_queue.pop(0)  # Get the next expert in line
            meeting.chats.append(self.Chat(
                agent=current_expert,
                content="I would like to use the shared resource to solve the task. I need 1 token to proceed. My reasoning score is {scores[current_expert.agent_name]}."
            ))
            
            # Check if resource is available
            if shared_resource['tokens'] > 0:
                shared_resource['tokens'] -= 1  # Deduct a token
                
                # Get answer from selected expert
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f"Please think step by step and then solve the task: {task}"
                ))
                
                expert_output = current_expert.forward(response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                })
                
                # Update the score based on output quality
                scores[current_expert.agent_name] += 1  # Increment score for successful output
                return expert_output["answer"]
            else:
                return "No resources available to answer the question."
        else:
            return "No requests to process."

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
