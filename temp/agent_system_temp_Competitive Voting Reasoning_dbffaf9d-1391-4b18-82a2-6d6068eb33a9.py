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
        agent_a = self.Agent(agent_name='Agent A', temperature=0.7)
        agent_b = self.Agent(agent_name='Agent B', temperature=0.7)
        voting_agent = self.Agent(agent_name='Voting Agent', temperature=0.5)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="competitive_voting")
        meeting.agents.extend([system, agent_a, agent_b, voting_agent])
        
        # Introduce the task and the competitive element
        meeting.chats.append(self.Chat(
            agent=system,
            content="We have a task to solve. Agent A and Agent B will present their arguments. After that, each agent will vote for their preferred answer."
        ))
        
        # Agent A presents their reasoning
        meeting.chats.append(self.Chat(
            agent=agent_a,
            content=f"I will solve the task: {task} by reasoning through my understanding step by step."
        ))
        agent_a_output = agent_a.forward(response_format={
            "thinking": "Your step by step reasoning.",
            "answer": "A single letter, A, B, C, or D."
        })
        meeting.chats.append(self.Chat(
            agent=agent_a,
            content=agent_a_output["thinking"] + agent_a_output["answer"]
        ))
        
        # Agent B presents their reasoning
        meeting.chats.append(self.Chat(
            agent=agent_b,
            content=f"I will counter Agent A's reasoning and solve the task: {task} with my own approach."
        ))
        agent_b_output = agent_b.forward(response_format={
            "thinking": "Your step by step reasoning.",
            "answer": "A single letter, A, B, C, or D."
        })
        meeting.chats.append(self.Chat(
            agent=agent_b,
            content=agent_b_output["thinking"] + agent_b_output["answer"]
        ))
        
        # Voting process initiated by the Voting Agent
        meeting.chats.append(self.Chat(
            agent=voting_agent,
            content="Now, based on your reasoning, please vote for your preferred answer."
        ))
        
        # Collect votes
        votes = {"A": 0, "B": 0, "C": 0, "D": 0}
        votes[agent_a_output["answer"]] += 1
        votes[agent_b_output["answer"]] += 1
        
        # Determine the final answer based on votes
        final_answer = max(votes, key=votes.get)
        # Handle tie situation
        if list(votes.values()).count(votes[final_answer]) > 1:
            final_answer = "TIE"  # or implement a tie-breaking mechanism
        
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
