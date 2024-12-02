import random
import numpy as np
import pandas

from mocked_base import MockAgent as Agent

from mocked_base import MockMeeting as Meeting

from mocked_base import MockChat as Chat

from sqlalchemy.orm import Session

class AgentSystem:
    def __init__(self, session: Session):
        self.Agent = Agent
        self.Meeting = Meeting
        self.Chat = Chat
        self.session = session

    def forward(self, task: str) -> str:
        # Create a system agent to provide instructions and evaluate arguments
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create two debate agents with opposing views
        agent_A = self.Agent(
            agent_name='Agent A',
            temperature=0.8
        )
        agent_B = self.Agent(
            agent_name='Agent B',
            temperature=0.8
        )
        
        # Setup meeting for the debate
        meeting = self.Meeting(meeting_name="scored_zero_sum_debate")
        meeting.agents.extend([system, agent_A, agent_B])
        
        # System instructs agents to argue for their answers
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please provide your reasoning for the task: {task}. Agent A, choose one answer and defend it. Agent B, choose a different answer and argue against Agent A's reasoning."
        ))
        
        # Get response from Agent A
        output_A = agent_A.forward(response_format={
            "thinking": "Your reasoning for the chosen answer.",
            "answer": "A single letter, A, B, C, or D."
        })
        meeting.chats.append(self.Chat(
            agent=agent_A,
            content=output_A["thinking"]
        ))
        
        # Get response from Agent B
        output_B = agent_B.forward(response_format={
            "thinking": "Your reasoning for the chosen answer, including counterarguments against Agent A.",
            "answer": "A single letter, A, B, C, or D."
        })
        meeting.chats.append(self.Chat(
            agent=agent_B,
            content=output_B["thinking"]
        ))
        
        # System evaluates the arguments from both agents
        score_A = system.forward(response_format={
            "evaluation": "Score the argument from Agent A based on clarity, logic, and relevance."
        })
        score_B = system.forward(response_format={
            "evaluation": "Score the argument from Agent B based on clarity, logic, and relevance."
        })
        
        # Determine the final answer based on scores
        if score_A > score_B:
            final_answer = output_A["answer"]
        else:
            final_answer = output_B["answer"]
        
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
