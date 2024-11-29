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
        debate_agent_1 = self.Agent(agent_name='Debate Agent 1', temperature=0.7)
        debate_agent_2 = self.Agent(agent_name='Debate Agent 2', temperature=0.7)
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.5)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="structured_debate_meeting")
        meeting.agents.extend([system, debate_agent_1, debate_agent_2, final_decision_agent])
        
        # Each debate agent presents their reasoning
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please present your reasoning for the task: {task}"
        ))
        
        # Debate Agent 1 presents reasoning
        debate_output_1 = debate_agent_1.forward(response_format={
            "thinking": "Your reasoning for the task.",
            "answer": "A single letter, A, B, C, or D."
        })
        meeting.chats.append(self.Chat(
            agent=debate_agent_1,
            content=debate_output_1["thinking"]
        ))
        
        # Debate Agent 2 presents reasoning
        debate_output_2 = debate_agent_2.forward(response_format={
            "thinking": "Your reasoning for the task.",
            "answer": "A single letter, A, B, C, or D."
        })
        meeting.chats.append(self.Chat(
            agent=debate_agent_2,
            content=debate_output_2["thinking"]
        ))
        
        # Facilitate structured debate and critique
        meeting.chats.append(self.Chat(
            agent=system,
            content="Now, please critique each other's reasoning and provide specific insights."
        ))
        
        # Debate Agent 1 critiques Debate Agent 2
        critique_output_1 = debate_agent_1.forward(response_format={
            "critique": "Your critique of the other agent's reasoning."
        })
        meeting.chats.append(self.Chat(
            agent=debate_agent_1,
            content=critique_output_1["critique"]
        ))
        
        # Debate Agent 2 critiques Debate Agent 1
        critique_output_2 = debate_agent_2.forward(response_format={
            "critique": "Your critique of the other agent's reasoning."
        })
        meeting.chats.append(self.Chat(
            agent=debate_agent_2,
            content=critique_output_2["critique"]
        ))
        
        # Allow agents to respond to critiques
        meeting.chats.append(self.Chat(
            agent=system,
            content="Now, please respond to each other's critiques."
        ))
        
        # Debate Agent 1 responds to Debate Agent 2's critique
        response_output_1 = debate_agent_1.forward(response_format={
            "response": "Your response to the other agent's critique."
        })
        meeting.chats.append(self.Chat(
            agent=debate_agent_1,
            content=response_output_1["response"]
        ))
        
        # Debate Agent 2 responds to Debate Agent 1's critique
        response_output_2 = debate_agent_2.forward(response_format={
            "response": "Your response to the other agent's critique."
        })
        meeting.chats.append(self.Chat(
            agent=debate_agent_2,
            content=response_output_2["response"]
        ))
        
        # Final decision based on the structured debate
        meeting.chats.append(self.Chat(
            agent=system,
            content="Given the debates and critiques, please provide a final answer to the task: {task}"
        ))
        
        final_output = final_decision_agent.forward(response_format={
            "final_thinking": "Your final reasoning considering the debate.",
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
