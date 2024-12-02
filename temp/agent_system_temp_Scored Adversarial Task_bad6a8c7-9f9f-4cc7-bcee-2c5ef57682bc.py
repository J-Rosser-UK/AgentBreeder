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
        # Create system and agent instances
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        agent_A = self.Agent(
            agent_name='Agent A',
            temperature=0.7
        )
        
        agent_B = self.Agent(
            agent_name='Agent B',
            temperature=0.7
        )
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="scored_adversarial_task")
        meeting.agents.extend([system, agent_A, agent_B])
        
        # Each agent presents their answer
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Agent A, please defend your answer to the task: {task}"
        ))
        
        output_A = agent_A.forward(response_format={
            "reasoning": "Your reasoning for the answer.",
            "answer": "A single letter, A, B, C or D."
        })
        
        meeting.chats.append(self.Chat(
            agent=agent_A,
            content=output_A["reasoning"]
        ))
        
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Agent B, please critique Agent A's reasoning and defend your own answer to the task: {task}"
        ))
        
        output_B = agent_B.forward(response_format={
            "reasoning": "Your reasoning for the answer.",
            "answer": "A single letter, A, B, C or D."
        })
        
        meeting.chats.append(self.Chat(
            agent=agent_B,
            content=output_B["reasoning"]
        ))
        
        # System evaluates both arguments
        meeting.chats.append(self.Chat(
            agent=system,
            content="Evaluating arguments from both agents based on clarity, relevance, and strength."
        ))
        
        score_A = system.forward(response_format={
            "score": "Score for Agent A's reasoning."
        })
        score_B = system.forward(response_format={
            "score": "Score for Agent B's reasoning."
        })
        
        # Determine final decision based on scores
        final_decision = "A" if score_A > score_B else "B"
        return final_decision

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
