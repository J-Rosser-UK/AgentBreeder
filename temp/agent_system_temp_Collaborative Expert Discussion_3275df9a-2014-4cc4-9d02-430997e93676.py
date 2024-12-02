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
        experts = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.5)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_expert_discussion")
        meeting.agents.extend([system] + list(experts.values()) + [final_decision_agent])
        
        # Each expert solves the task
        expert_outputs = []
        for expert in experts.values():
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"{expert.agent_name}, please think step by step and solve the task: {task}"
            ))
            output = expert.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C or D."
            })
            meeting.chats.append(self.Chat(
                agent=expert,
                content=output["thinking"] + output["answer"]
            ))
            expert_outputs.append(output["answer"])
        
        # Facilitate discussion among experts
        discussion_content = "Experts, please discuss your answers and reasoning."
        meeting.chats.append(self.Chat(
            agent=system,
            content=discussion_content
        ))
        
        # Final decision based on expert inputs
        meeting.chats.append(self.Chat(
            agent=system,
            content="Given the responses from all experts, please evaluate and provide the best answer."
        ))
        final_output = final_decision_agent.forward(response_format={
            "expert_answers": expert_outputs
        })
        
        return final_output["final_answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
