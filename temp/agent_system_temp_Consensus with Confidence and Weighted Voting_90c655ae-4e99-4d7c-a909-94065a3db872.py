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
        meeting = self.Meeting(meeting_name="consensus_with_confidence_meeting")
        meeting.agents.extend([system, principle_agent, cot_agent_1, cot_agent_2])
        
        # Get the principles involved from the principle agent
        meeting.chats.append(self.Chat(
            agent=system,
            content="What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
        ))
        
        principle_output = principle_agent.forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        
        meeting.chats.append(self.Chat(
            agent=principle_agent,
            content=principle_output["thinking"] + principle_output["principles"]
        ))
        
        # Now solve using the principles with both cot agents
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the question and the involved principles above, think step by step and then solve the task: {task}"
        ))
        
        max_turns = 2  # Limit the number of turns each agent can use
        answers = []
        confidences = []
        
        for agent in [cot_agent_1, cot_agent_2]:
            for _ in range(max_turns):
                final_output = agent.forward(response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D.",
                    "confidence": "Your confidence level (0-1)."
                })
                answers.append(final_output["answer"])
                confidences.append(final_output["confidence"])
        
        # Aggregate answers from both cot agents by implementing a weighted voting mechanism
        answer_weights = {answer: 0 for answer in set(answers)}
        for answer, confidence in zip(answers, confidences):
            answer_weights[answer] += confidence
        
        # Determine the final answer with the highest weighted score
        final_answer = max(answer_weights, key=answer_weights.get)
        
        # Return the final answer based on the weighted voting
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
