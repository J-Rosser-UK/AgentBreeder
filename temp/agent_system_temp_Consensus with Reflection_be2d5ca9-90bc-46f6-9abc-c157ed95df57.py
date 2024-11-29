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
        meeting = self.Meeting(meeting_name="consensus_with_reflection_meeting")
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
        
        # Ensure confidence values are floats
        confidence_1 = float(final_output_1["confidence"])
        confidence_2 = float(final_output_2["confidence"])
        
        # Aggregate answers from both cot agents by implementing a voting mechanism
        answers = [final_output_1["answer"], final_output_2["answer"]]
        confidences = [confidence_1, confidence_2]
        
        # Weighted voting based on confidence
        weighted_votes = {"A": 0, "B": 0, "C": 0, "D": 0}
        for answer, confidence in zip(answers, confidences):
            weighted_votes[answer] += confidence
        
        # Determine the final answer based on the highest weighted vote
        final_answer = max(weighted_votes, key=weighted_votes.get)
        
        # Reflection phase: allow agents to revise answers
        meeting.chats.append(self.Chat(
            agent=system,
            content="Based on the answers and confidence levels provided, please reflect on your response and revise if necessary."
        ))
        
        # Allow agents to reflect and potentially revise their answers
        for agent in [cot_agent_1, cot_agent_2]:
            reflection_output = agent.forward(response_format={
                "thinking": "Reflect on your previous answer and provide a revised one if needed.",
                "answer": "A single letter, A, B, C or D.",
                "confidence": "Your confidence level (0-1)."
            })
            meeting.chats.append(self.Chat(
                agent=agent,
                content=reflection_output["thinking"] + reflection_output["answer"]
            ))
        
        # Final aggregation of revised answers
        final_answers = [reflection_output["answer"] for reflection_output in [final_output_1, final_output_2]]
        final_confidences = [float(reflection_output["confidence"]) for reflection_output in [final_output_1, final_output_2]]
        final_weighted_votes = {"A": 0, "B": 0, "C": 0, "D": 0}
        for answer, confidence in zip(final_answers, final_confidences):
            final_weighted_votes[answer] += confidence
        
        # Determine the final answer based on the highest weighted vote after reflection
        final_answer_reflected = max(final_weighted_votes, key=final_weighted_votes.get)
        
        return final_answer_reflected

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
