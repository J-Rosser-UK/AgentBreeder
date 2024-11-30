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
        principle_agent = self.Agent(agent_name='Principle Agent', temperature=0.8)
        cot_agent_1 = self.Agent(agent_name='Chain-of-Thought Agent 1', temperature=0.8)
        cot_agent_2 = self.Agent(agent_name='Chain-of-Thought Agent 2', temperature=0.8)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name='structured_collaborative_memory_meeting')
        meeting.agents.extend([system, principle_agent, cot_agent_1, cot_agent_2])
        
        # Initialize a shared memory buffer with tagging system
        shared_memory = {}  # Using a dictionary for tagging principles
        
        # Get the principles involved from the principle agent
        meeting.chats.append(self.Chat(
            agent=system,
            content="What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
        ))
        
        principle_output = principle_agent.forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        
        # Store principles in shared memory with tags
        shared_memory['principles'] = principle_output["principles"]
        meeting.chats.append(self.Chat(
            agent=principle_agent,
            content=principle_output["thinking"] + principle_output["principles"]
        ))
        
        # Now solve using the principles with both cot agents
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the question and the involved principles above, think step by step and then solve the task: {task}. Use the shared memory for insights: {shared_memory['principles']}"
        ))
        
        final_output_1 = cot_agent_1.forward(response_format={
            "thinking": "Your step by step thinking referencing shared principles.",
            "answer": "A single letter, A, B, C or D.",
            "confidence": "Your confidence level (0-1)."
        })
        final_output_2 = cot_agent_2.forward(response_format={
            "thinking": "Your step by step thinking referencing shared principles.",
            "answer": "A single letter, A, B, C or D.",
            "confidence": "Your confidence level (0-1)."
        })
        
        # Aggregate answers from both cot agents by implementing a voting mechanism
        answers = [final_output_1["answer"], final_output_2["answer"]]
        confidences = [final_output_1["confidence"], final_output_2["confidence"]]
        final_answer = max(set(answers), key=answers.count)  # Majority voting
        
        # Return the final answer based on the majority
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
