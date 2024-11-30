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
        cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="structured_knowledge_base_meeting")
        meeting.agents.extend([system, cot_agent, final_decision_agent])
        
        # Define a structured knowledge base
        knowledge_base = {
            "constellations": [
                {"name": "Cassiopeia", "description": "A bright W-shaped constellation in the northern sky."},
                {"name": "Centaurus", "description": "A prominent constellation in the southern sky."},
                {"name": "Cygnus", "description": "A constellation named after the Latin word for swan."},
                {"name": "Cepheus", "description": "A constellation representing a king in Greek mythology."}
            ]
        }
        
        N_max = 3  # Maximum number of attempts
        
        # Initial attempt
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Solve: {task} step by step."
        ))
        
        output = cot_agent.forward(response_format={
            "thinking": "Your reasoning.",
            "answer": "A, B, C, or D."
        })
        
        # Iterate and refine the answer
        for _ in range(N_max):
            # Request specific knowledge from the knowledge base
            meeting.chats.append(self.Chat(
                agent=cot_agent,
                content=f"I want to access knowledge about constellations to help solve the task. My reasoning is: {output['thinking']}"
            ))
            request_output = cot_agent.forward(response_format={
                "request": "Your justification for accessing the knowledge base."
            })
            
            # Evaluate the request
            if request_output['request']:
                # Grant access to the knowledge base
                relevant_knowledge = [k for k in knowledge_base['constellations'] if k['name'] in output['thinking']]
                meeting.chats.append(self.Chat(
                    agent=system,
                    content="Access granted to the knowledge base."
                ))
            else:
                meeting.chats.append(self.Chat(
                    agent=system,
                    content="Access denied. Please improve your reasoning."
                ))
                continue  # Skip to the next iteration if access is denied
            
            # Use the knowledge from the base to refine the answer
            meeting.chats.append(self.Chat(
                agent=cot_agent,
                content=f"Refine your approach for: {task} using the knowledge: {relevant_knowledge}."
            ))
            output = cot_agent.forward(response_format={
                "thinking": "Your refined reasoning.",
                "answer": "A, B, C, or D."
            })
        
        # Make final decision
        meeting.chats.append(self.Chat(
            agent=system,
            content="Final answer based on all solutions and knowledge base contributions."
        ))
        
        final_output = final_decision_agent.forward(response_format={
            "thinking": "Your reasoning based on knowledge base and all solutions.",
            "answer": "A, B, C, or D."
        })
        
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
