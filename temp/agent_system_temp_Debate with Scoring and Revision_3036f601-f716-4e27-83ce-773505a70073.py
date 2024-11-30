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
        principle_agent_1 = self.Agent(agent_name='Principle Agent 1', temperature=0.8)
        principle_agent_2 = self.Agent(agent_name='Principle Agent 2', temperature=0.8)
        cot_agent_1 = self.Agent(agent_name='Chain-of-Thought Agent 1', temperature=0.8)
        cot_agent_2 = self.Agent(agent_name='Chain-of-Thought Agent 2', temperature=0.8)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="debate_consensus_meeting")
        meeting.agents.extend([system, principle_agent_1, principle_agent_2, cot_agent_1, cot_agent_2])
        
        # First get the principles involved from both principle agents
        meeting.chats.append(self.Chat(
            agent=system,
            content="What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
        ))
        
        principle_output_1 = principle_agent_1.forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        principle_output_2 = principle_agent_2.forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        
        meeting.chats.append(self.Chat(
            agent=principle_agent_1,
            content=principle_output_1["thinking"] + principle_output_1["principles"]
        ))
        meeting.chats.append(self.Chat(
            agent=principle_agent_2,
            content=principle_output_2["thinking"] + principle_output_2["principles"]
        ))
        
        # Now solve using the principles with both cot agents
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the question and the involved principles above, think step by step and then solve the task: {task}"
        ))
        
        final_output_1 = cot_agent_1.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D.",
            "confidence": "Your confidence level (0-1)."
        })
        final_output_2 = cot_agent_2.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D.",
            "confidence": "Your confidence level (0-1)."
        })
        
        # Debate phase: each agent critiques the other's answer with scoring
        meeting.chats.append(self.Chat(
            agent=system,
            content="Now let's have a debate. Each agent will critique the other’s answer and provide a score from 1 to 10."
        ))
        debate_output_1 = cot_agent_1.forward(response_format={
            "critique": "Your critique of Agent 2's answer.",
            "score": "Your score for Agent 2's answer (1-10)."
        })
        debate_output_2 = cot_agent_2.forward(response_format={
            "critique": "Your critique of Agent 1's answer.",
            "score": "Your score for Agent 1's answer (1-10)."
        })
        
        meeting.chats.append(self.Chat(
            agent=cot_agent_1,
            content=debate_output_1["critique"]
        ))
        meeting.chats.append(self.Chat(
            agent=cot_agent_2,
            content=debate_output_2["critique"]
        ))
        
        # Final decision based on the critiques and confidence levels
        final_answer = final_output_1["answer"] if final_output_1["confidence"] + debate_output_2["score"] > final_output_2["confidence"] + debate_output_1["score"] else final_output_2["answer"]
        
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
