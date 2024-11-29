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
        meeting = self.Meeting(meeting_name="collaborative_reasoning_meeting")
        meeting.agents.extend([system, principle_agent, cot_agent_1, cot_agent_2])
        
        # Get the principles involved from the principle agent
        meeting.chats.append(self.Chat(
            agent=system,
            content="Please analyze the task and identify relevant principles that can aid in solving it. Provide a detailed explanation of each principle."
        ))
        
        principle_output = principle_agent.forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        
        meeting.chats.append(self.Chat(
            agent=principle_agent,
            content=principle_output["thinking"] + principle_output["principles"]
        ))
        
        # Engage in dialogue between CoT agents
        for cot_agent in [cot_agent_1, cot_agent_2]:
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Based on the identified principles, {cot_agent.agent_name}, please think step by step about the task: '{task}'. What are your thoughts?"
            ))
            cot_output = cot_agent.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D.",
                "confidence": "Your confidence level (0-1)."
            })
            meeting.chats.append(self.Chat(
                agent=cot_agent,
                content=cot_output["thinking"] + f" My answer is: {cot_output["answer"]} with confidence {cot_output["confidence"]}."
            ))
        
        # Collect answers and confidence levels from both CoT agents
        answers = [cot_agent_1.forward(response_format={"answer": "A single letter, A, B, C or D."}), cot_agent_2.forward(response_format={"answer": "A single letter, A, B, C or D."})]
        confidences = [cot_agent_1.forward(response_format={"confidence": "Your confidence level (0-1)."}), cot_agent_2.forward(response_format={"confidence": "Your confidence level (0-1)."})]
        
        # Implement weighted voting mechanism
        final_answer = weighted_voting(answers, confidences)
        
        # Return the final answer
        return final_answer
    
    
    def weighted_voting(answers, confidences):
        weighted_counts = {}
        for answer, confidence in zip(answers, confidences):
            if answer not in weighted_counts:
                weighted_counts[answer] = 0
            weighted_counts[answer] += confidence
        return max(weighted_counts, key=weighted_counts.get)

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
