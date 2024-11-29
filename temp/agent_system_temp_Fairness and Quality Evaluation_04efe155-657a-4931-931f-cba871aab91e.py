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
        
        # Create multiple agents with an emphasis on diverse reasoning
        N = 3  # Number of agents
        agents = [
            self.Agent(
                agent_name=f'Diverse Reasoning Agent {i}',
                temperature=0.8
            ) for i in range(N)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="fairness_quality_evaluation")
        meeting.agents.extend([system] + agents)
        
        # Collect answers and reasoning scores from all agents
        answers = []
        reasoning_scores = []
        for i in range(N):
            # Add system instruction
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Please think step by step, consider multiple perspectives, and solve the task: {task}"
                )
            )
            
            # Get response from current agent
            output = agents[i].forward(
                response_format={
                    "thinking": "Your step by step reasoning considering fairness.",
                    "answer": "A single letter, A, B, C, or D.",
                    "score": "A score for the quality of your reasoning (1-10)"
                }
            )
            
            # Record the agent's response and reasoning score
            meeting.chats.append(
                self.Chat(
                    agent=agents[i], 
                    content=output["thinking"]
                )
            )
            answers.append(output["answer"])
            reasoning_scores.append(int(output["score"]))
        
        # Evaluate answers by considering reasoning scores
        from collections import Counter
        answer_counts = Counter(answers)
        most_common_answer = answer_counts.most_common(1)[0][0]
        
        # Calculate total reasoning score for the most common answer
        total_score = sum(score for answer, score in zip(answers, reasoning_scores) if answer == most_common_answer)
        
        # Return the most common answer only if it has a sufficient quality score
        threshold = 5  # Define a threshold for quality
        return most_common_answer if total_score >= threshold else None
    

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
