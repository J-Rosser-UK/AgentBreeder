import random
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
        meeting = self.Meeting(meeting_name="fairness-driven-scoring")
        meeting.agents.extend([system] + cot_agents)
        
        # Collect answers and reasoning from all agents
        possible_answers = []
        reasoning_map = []
        for agent in cot_agents:
            # Add system instruction
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Please think step by step and then solve the task: {task}"
                )
            )
            
            # Get response from current COT agent
            output = agent.forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            
            # Record the agent's response
            meeting.chats.append(
                self.Chat(
                    agent=agent, 
                    content=output["thinking"]
                )
            )
            
            possible_answers.append(output["answer"])
            reasoning_map.append(output["thinking"])
        
        # Scoring mechanism
        from collections import Counter
        answer_counts = Counter(possible_answers)
        scores = {answer: 0 for answer in answer_counts.keys()}
        
        for answer in answer_counts:
            scores[answer] += answer_counts[answer]  # Majority score
            scores[answer] += len([reason for reason in reasoning_map if answer in reason])  # Diversity score
        
        # Determine final answer based on highest score
        final_answer = max(scores, key=scores.get)
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
