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
        # Create a system agent to provide instructions
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create multiple CoT agents with higher temperature for varied reasoning
        N = 3  # Number of CoT agents
        cot_agents = [
            self.Agent(
                agent_name=f'Chain-of-Thought Agent {i}',
                temperature=0.8
            ) for i in range(N)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="weighted_self_consistency")
        meeting.agents.extend([system] + cot_agents)
        
        # Collect answers and confidence ratings from all agents
        answers_with_confidence = []
        for i in range(N):
            # Add concise system instruction
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Please solve this task: {task}"
                )
            )
            
            # Get response from current COT agent
            output = cot_agents[i].forward(
                response_format={
                    "thinking": "Your reasoning.",
                    "answer": "A, B, C, or D.",
                    "confidence": "Your confidence score (0-1)"
                }
            )
            
            # Record the agent's response with confidence
            meeting.chats.append(
                self.Chat(
                    agent=cot_agents[i], 
                    content=output["thinking"]
                )
            )
            
            # Append answer and confidence to the list
            answers_with_confidence.append((output["answer"], output["confidence"]))
        
        # Implement weighted voting based on confidence
        weighted_answers = {}  # Dictionary to hold weighted counts
        for answer, confidence in answers_with_confidence:
            weighted_answers[answer] = weighted_answers.get(answer, 0) + confidence  # Weighting by confidence
        
        # Select the answer with the highest weighted score
        final_answer = max(weighted_answers, key=weighted_answers.get)
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
