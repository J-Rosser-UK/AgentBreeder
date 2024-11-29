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

    def forward(self, task: str, correct_answer: str) -> str:
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        leader = self.Agent(agent_name='Leading Agent', temperature=0.9)
        regular_agents = [
            self.Agent(agent_name=f'Regular Agent {i}', temperature=0.7) for i in range(3)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name='cooperative_coalition_meeting')
        meeting.agents.extend([system, leader] + regular_agents)
        
        # Initial instruction to agents
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please assess the task: {task} and discuss your reasoning with each other."
        ))
        
        # Leader assesses the task and guides the discussion
        leader_output = leader.forward(response_format={
            "reasoning": "Your assessment of the task and guidance for the discussion.",
            "confidence": "Your confidence level (0-1)"
        })
        meeting.chats.append(self.Chat(
            agent=leader,
            content=leader_output["reasoning"] + f" (Confidence: {leader_output['confidence']})"
        ))
        
        coalition_agents = []
        for agent in regular_agents:
            output = agent.forward(response_format={
                "reasoning": "Your reasoning for forming a coalition or going solo.",
                "confidence": "Your confidence level (0-1)"
            })
            meeting.chats.append(self.Chat(
                agent=agent,
                content=output["reasoning"] + f" (Confidence: {output['confidence']})"
            ))
            # Validate confidence level
            try:
                confidence = float(output["confidence"])
                if confidence >= 0.5:
                    coalition_agents.append((agent, confidence))
                else:
                    coalition_agents.append((agent, 0.5))  # Default confidence level if below threshold
            except ValueError:
                coalition_agents.append((agent, 0.5))  # Default confidence level for invalid inputs
        
        # Reward shaping: Initialize rewards
        rewards = {agent: 0 for agent in regular_agents}
        
        # If no coalitions were formed, proceed with individual answers
        if not coalition_agents:
            answers = []
            for agent in regular_agents:
                final_output = agent.forward(response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C, or D."
                })
                answers.append(final_output["answer"])
            final_answer = max(set(answers), key=answers.count)
        else:
            # Coalition formed, agents collaborate under leader's guidance
            coalition_answers = []
            total_confidence = 0
            for agent, confidence in coalition_agents:
                final_output = agent.forward(response_format={
                    "thinking": "Your collaborative reasoning guided by the leader.",
                    "answer": "A single letter, A, B, C, or D."
                })
                coalition_answers.append((final_output["answer"], confidence))
                total_confidence += confidence
                rewards[agent] += 1  # Reward for collaboration
            
            # Weighted voting based on confidence
            weighted_votes = {}
            for answer, confidence in coalition_answers:
                if answer not in weighted_votes:
                    weighted_votes[answer] = 0
                weighted_votes[answer] += confidence
            final_answer = max(weighted_votes, key=weighted_votes.get)
        
        # Correctness evaluation
        for agent in regular_agents:
            if final_answer == correct_answer:
                rewards[agent] += 2  # Reward for correct answers
            else:
                rewards[agent] -= 1  # Penalty for incorrect answers
        
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
