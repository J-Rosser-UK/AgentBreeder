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

    def evaluate_reasoning(reasoning: str) -> int:
        # Simple scoring function based on reasoning length
        return len(reasoning.split())  # Score based on the number of words in reasoning
    
    
    def forward(self, task: str) -> str:
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        leader = self.Agent(agent_name='Leading Agent', temperature=0.9)
        regular_agents = [
            self.Agent(agent_name=f'Regular Agent {i}', temperature=0.7) for i in range(3)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name='zero_sum_competition_meeting')
        meeting.agents.extend([system, leader] + regular_agents)
        
        # Initial instruction to agents
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please assess the task: {task}. Remember, this is a zero-sum game where your success depends on outsmarting the others."
        ))
        
        # Leader assesses the task and guides the discussion
        leader_output = leader.forward(response_format={
            "reasoning": "Your assessment of the task and guidance for the competition.",
            "confidence": "Your confidence level (0-1)"
        })
        meeting.chats.append(self.Chat(
            agent=leader,
            content=leader_output["reasoning"] + f" (Confidence: {leader_output['confidence']})"
        ))
        
        # Each regular agent competes and provides their reasoning
        answers = []
        scores = []
        for agent in regular_agents:
            output = agent.forward(response_format={
                "reasoning": "Your reasoning for your answer.",
                "answer": "A single letter, A, B, C, or D."
            })
            meeting.chats.append(self.Chat(
                agent=agent,
                content=output.get("reasoning", "No reasoning provided.") + f" Answer: {output.get('answer', 'No answer provided.') }"
            ))
            answers.append(output.get("answer", 'D'))  # Default to 'D' if no answer provided
            score = evaluate_reasoning(output.get("reasoning", ''))  # Handle empty reasoning
            scores.append(score)
        
        # Determine the winner based on the responses and scores
        final_answer = max(set(answers), key=answers.count)
        final_score = max(scores)
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
