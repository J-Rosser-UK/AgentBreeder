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
        # Create a supervisor agent
        supervisor = self.Agent(agent_name='Supervisor Agent', temperature=0.8)
        # Create specialized agents
        agents = [self.Agent(agent_name=f'Agent {i}', temperature=0.8) for i in range(5)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name='negotiated_coalition_meeting')
        meeting.agents.extend([supervisor] + agents)
        
        # Supervisor prompts agents to propose coalitions
        meeting.chats.append(self.Chat(
            agent=supervisor,
            content="Based on your expertise, propose a coalition with 1 or 2 other agents to solve the task: {task}."
        ))
        
        coalitions = []
        for agent in agents:
            coalition_output = agent.forward(response_format={
                "coalition": "Names of agents you want to form a coalition with."
            })
            coalitions.append((agent.agent_name, coalition_output["coalition"]))
            meeting.chats.append(self.Chat(
                agent=agent,
                content=f"I propose to form a coalition with: {coalition_output['coalition']}.",
            ))
        
        # Allow coalitions to form and negotiate
        coalition_groups = {}  # Dictionary to hold coalitions
        for agent_name, coalition in coalitions:
            for member in coalition.split(', '):
                if member not in coalition_groups:
                    coalition_groups[member] = []
                coalition_groups[member].append(agent_name)
        
        # Negotiation phase for coalitions
        for coalition_leader, members in coalition_groups.items():
            meeting.chats.append(self.Chat(
                agent=supervisor,
                content=f"Coalition led by {coalition_leader}, please discuss and adjust your coalition if needed."
            ))
            # Allow agents to negotiate within the coalition
            for member in members:
                negotiation_output = self.Agent(agent_name=member, temperature=0.8).forward(response_format={
                    "negotiation": "Discuss if adjustments to the coalition are needed."
                })
                meeting.chats.append(self.Chat(
                    agent=member,
                    content=negotiation_output["negotiation"]
                ))
    
        # Each coalition will reason together
        answers = []
        for coalition_leader, members in coalition_groups.items():
            meeting.chats.append(self.Chat(
                agent=supervisor,
                content=f"Coalition led by {coalition_leader}, please reason step by step to solve the task: {task}."
            ))
            coalition_output = self.Agent(agent_name=coalition_leader, temperature=0.8).forward(response_format={
                "thinking": "Your step by step reasoning as a coalition.",
                "answer": "A single letter, A, B, C, or D."
            })
            answers.append(coalition_output["answer"])
            meeting.chats.append(self.Chat(
                agent=coalition_leader,
                content=coalition_output["thinking"] + coalition_output["answer"]
            ))
        
        # Voting mechanism for final answer
        from collections import Counter
        final_answer = Counter(answers).most_common(1)[0][0]
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
