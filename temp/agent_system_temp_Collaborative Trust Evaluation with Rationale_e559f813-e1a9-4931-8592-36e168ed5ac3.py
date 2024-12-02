import random
import numpy as np
import pandas

from mocked_base import MockAgent as Agent

from mocked_base import MockMeeting as Meeting

from mocked_base import MockChat as Chat

from sqlalchemy.orm import Session

class AgentSystem:
    def __init__(self, session: Session = None):
        self.Agent = Agent
        self.Meeting = Meeting
        self.Chat = Chat
        self.session = session

    def forward(self, task: str) -> str:
        # Create a system agent to provide instructions
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create multiple CoT agents for varied reasoning
        N = 3  # Number of CoT agents
        cot_agents = [
            self.Agent(
                agent_name=f'Chain-of-Thought Agent {i}',
                temperature=0.8
            ) for i in range(N)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_trust")
        meeting.agents.extend([system] + cot_agents)
        
        # Collect answers and trust scores
        possible_answers = []
        trust_scores = []
        rationales = []
        for i in range(N):
            # Add system instruction
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Please think step by step and then solve the task: {task}"
                )
            )
            
            # Get response from current COT agent
            output = cot_agents[i].forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D.",
                    "rationale": "Explain why you trust this answer."
                }
            )
            
            # Record the agent's response
            meeting.chats.append(
                self.Chat(
                    agent=cot_agents[i], 
                    content=output["thinking"]
                )
            )
            
            possible_answers.append(output["answer"])
            trust_scores.append(evaluate_trust(output["answer"], output["rationale"]))
            rationales.append(output["rationale"])
        
        # Weighted voting based on trust scores
        final_answer = weighted_voting(possible_answers, trust_scores)
        return final_answer
    
    
    def evaluate_trust(answer, rationale):
        # Implement logic to evaluate trust based on answer and rationale
        return 1.0  # Placeholder for actual logic
    
    
    def weighted_voting(answers, trust_scores):
        from collections import Counter
        weighted_counts = Counter()
        for answer, trust in zip(answers, trust_scores):
            weighted_counts[answer] += trust
        return weighted_counts.most_common(1)[0][0]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
