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
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="competitive_reasoning_meeting")
        meeting.agents.extend([system, agent_a, agent_b])
        
        # Introduce the task and the zero-sum nature
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"You are in a competitive scenario. Agent A aims to promote answer A, while Agent B aims to promote answer B. The task is: {task}. Each agent will present their reasoning and strategy to convince others why their answer is correct."
        ))
        
        # Agent A reasoning
        meeting.chats.append(self.Chat(
            agent=agent_a,
            content="Please reason step by step why answer A is the best choice for the task."
        ))
        output_a = agent_a.forward(response_format={
            "thinking": "Your reasoning for answer A.",
            "answer": "A single letter, A.",
            "score": "Score for reasoning quality (1-10)"
        })
        
        # Agent B reasoning
        meeting.chats.append(self.Chat(
            agent=agent_b,
            content="Please reason step by step why answer B is the best choice for the task."
        ))
        output_b = agent_b.forward(response_format={
            "thinking": "Your reasoning for answer B.",
            "answer": "A single letter, B.",
            "score": "Score for reasoning quality (1-10)"
        })
        
        # Critique phase for Agent A
        meeting.chats.append(self.Chat(
            agent=agent_a,
            content=f"Critique Agent B's reasoning: {output_b['thinking']}"
        ))
        critique_a = agent_a.forward(response_format={
            "critique": "Your critique of Agent B's reasoning."
        })
        
        # Critique phase for Agent B
        meeting.chats.append(self.Chat(
            agent=agent_b,
            content=f"Critique Agent A's reasoning: {output_a['thinking']}"
        ))
        critique_b = agent_b.forward(response_format={
            "critique": "Your critique of Agent A's reasoning."
        })
        
        # Final decision based on the reasoning and scores
        final_decision_content = f"Based on the reasoning and critiques provided by both agents, please determine the final answer considering the scores: A: {output_a['score']}, B: {output_b['score']}.
    "
        meeting.chats.append(self.Chat(
            agent=system,
            content=final_decision_content
        ))
        final_decision = system.forward(response_format={
            "thinking": "Your analysis of both arguments and critiques.",
            "answer": "A single letter, A or B."
        })
        
        return final_decision["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
