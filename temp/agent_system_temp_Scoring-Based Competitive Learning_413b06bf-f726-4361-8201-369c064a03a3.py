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
        # Create system and agent instances
        system = self.Agent(agent_name='system', temperature=0.8)
        cot_agents = [self.Agent(agent_name=f'Chain-of-Thought Agent {i+1}', temperature=0.7) for i in range(4)]
        critic = self.Agent(agent_name='Critic Agent', temperature=0.6)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="scoring_competition")
        meeting.agents.extend([system] + cot_agents + [critic])
        
        # Define a limited shared resource (tokens)
        shared_tokens = 100  # Total tokens available for all agents
        outputs = []
        scores = []
        
        # Initial attempts
        for cot_agent in cot_agents:
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"You have {shared_tokens // len(cot_agents)} tokens to think step by step and solve the task: {task}"
                )
            )
            output = cot_agent.forward(
                response_format={
                    "thinking": "Your step by step reasoning using limited tokens.",
                    "answer": "A single letter, A, B, C, or D."
                }
            )
            outputs.append(output)
            meeting.chats.append(
                self.Chat(
                    agent=cot_agent, 
                    content=output["thinking"]
                )
            )
        
        # Critique phase
        meeting.chats.append(
            self.Chat(
                agent=system, 
                content="Please review the answers above and provide scores for each response based on clarity, correctness, and creativity."
            )
        )
        critic_output = critic.forward(
            response_format={
                "scores": "Scores for each Chain-of-Thought Agent's response. Provide a JSON array of numerical values."
            }
        )
        # Ensure scores are numeric
        try:
            scores = [int(score) for score in critic_output["scores"]]  # Expecting a JSON array of numbers
        except (ValueError, TypeError):
            raise ValueError("Critic output is not in the expected format. Please check the critic's response.")
        meeting.chats.append(
            self.Chat(
                agent=critic, 
                content="Scores assigned: " + str(scores)
            )
        )
        
        # Reflection and refinement based on feedback
        for j, cot_agent in enumerate(cot_agents):
            tokens_for_agent = max(1, scores[j])  # Ensure each agent has at least 1 token
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Given your score of {scores[j]}, refine your reasoning and try again: {task}"
                )
            )
            output = cot_agent.forward(
                response_format={
                    "thinking": "Your refined step by step thinking with limited tokens.",
                    "answer": "A single letter, A, B, C, or D."
                }
            )
            meeting.chats.append(
                self.Chat(
                    agent=cot_agent, 
                    content=output["thinking"]
                )
            )
        
        # Select the final answer based on consensus of the outputs
        answers = [output["answer"] for output in outputs]
        from collections import Counter
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common()  # Get the most common answer
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            return {"consensus": None, "status": "Ambiguous"}
        else:
            return {"consensus": most_common[0][0], "status": "Consensus"}

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
