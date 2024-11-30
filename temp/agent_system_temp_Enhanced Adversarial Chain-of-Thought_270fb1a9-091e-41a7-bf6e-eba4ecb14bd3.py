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
        cot_agent_1 = self.Agent(agent_name='Chain-of-Thought Agent 1', temperature=0.8)
        cot_agent_2 = self.Agent(agent_name='Chain-of-Thought Agent 2', temperature=0.8)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="enhanced_adversarial_chain_of_thought_meeting")
        meeting.agents.extend([system, cot_agent_1, cot_agent_2])
        
        # System instructs both agents to argue their case
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"You are both tasked with arguing your answer to the question: {task}. Present your reasoning and try to convince the other agent why your answer is correct."
        ))
        
        # Each agent presents their reasoning
        output_1 = cot_agent_1.forward(response_format={
            "thinking": "Your reasoning for your answer.",
            "answer": "A single letter, A, B, C or D."
        })
        meeting.chats.append(self.Chat(
            agent=cot_agent_1,
            content=output_1["thinking"]
        ))
        
        output_2 = cot_agent_2.forward(response_format={
            "thinking": "Your reasoning for your answer.",
            "answer": "A single letter, A, B, C or D."
        })
        meeting.chats.append(self.Chat(
            agent=cot_agent_2,
            content=output_2["thinking"]
        ))
        
        # Scoring mechanism to evaluate arguments
        score_1 = self.evaluate_argument(output_1["thinking"])
        score_2 = self.evaluate_argument(output_2["thinking"])
        
        # Determine the winning agent based on score
        if score_1 > score_2:
            return output_1["answer"]
        elif score_2 > score_1:
            return output_2["answer"]
        else:
            # Tie-breaking mechanism
            meeting.chats.append(self.Chat(
                agent=system,
                content="Both arguments are equally compelling. Please refine your arguments and provide a new answer."
            ))
            refined_output_1 = cot_agent_1.forward(response_format={
                "thinking": "Refine your reasoning based on the previous arguments.",
                "answer": "A single letter, A, B, C or D."
            })
            refined_output_2 = cot_agent_2.forward(response_format={
                "thinking": "Refine your reasoning based on the previous arguments.",
                "answer": "A single letter, A, B, C or D."
            })
            return refined_output_1["answer"] if self.evaluate_argument(refined_output_1["thinking"]) > self.evaluate_argument(refined_output_2["thinking"]) else refined_output_2["answer"]
    
        def evaluate_argument(self, reasoning: str) -> int:
            # Evaluate the clarity and relevance of the argument
            score = 0
            if "because" in reasoning:
                score += 2  # Award points for reasoning that includes justification
            if len(reasoning.split()) > 10:
                score += 1  # Award points for longer, potentially more detailed arguments
            # Add more evaluation criteria as needed
            return score

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
