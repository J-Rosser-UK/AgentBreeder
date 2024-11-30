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
        meeting = self.Meeting(meeting_name="clarifying_questions_meeting")
        meeting.agents.extend([system, principle_agent, cot_agent_1, cot_agent_2])
        
        # Ask clarifying questions
        meeting.chats.append(self.Chat(
            agent=system,
            content="What questions do you have about the task?"
        ))
        
        clarifying_output = principle_agent.forward(response_format={
            "questions": "Any clarifying questions you have."
        })
        meeting.chats.append(self.Chat(
            agent=principle_agent,
            content=clarifying_output["questions"]
        ))
        
        # Get principles concisely
        meeting.chats.append(self.Chat(
            agent=system,
            content="List principles relevant to this task."
        ))
        
        principle_output = principle_agent.forward(response_format={
            "thinking": "Your reasoning about the principles.",
            "principles": "List of principles."
        })
        meeting.chats.append(self.Chat(
            agent=principle_agent,
            content=principle_output["thinking"] + principle_output["principles"]
        ))
        
        # Solve using principles
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Solve the task: {task} using the principles listed."
        ))
        
        final_output_1 = cot_agent_1.forward(response_format={
            "thinking": "Your reasoning.",
            "answer": "A single letter, A, B, C, or D.",
            "confidence": "Your confidence level (0-1) as a float."
        })
        final_output_2 = cot_agent_2.forward(response_format={
            "thinking": "Your reasoning.",
            "answer": "A single letter, A, B, C, or D.",
            "confidence": "Your confidence level (0-1) as a float."
        })
        
        # Ensure confidence is float
        try:
            answers = [final_output_1["answer"], final_output_2["answer"]]
            confidences = [float(final_output_1["confidence"]), float(final_output_2["confidence"])]
        except ValueError:
            # Handle unexpected confidence formats
            raise ValueError("Confidence values must be numeric.")
        
        # Aggregate answers via weighted voting
        weighted_answers = {answer: 0 for answer in set(answers)}
        for answer, confidence in zip(answers, confidences):
            weighted_answers[answer] += confidence
        final_answer = max(weighted_answers, key=weighted_answers.get)  # Weighted voting
        
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
