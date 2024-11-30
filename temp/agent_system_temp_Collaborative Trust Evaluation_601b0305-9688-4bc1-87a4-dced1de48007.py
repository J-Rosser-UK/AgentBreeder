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
        cot_agents = [
            self.Agent(agent_name=f'Chain-of-Thought Agent {i}', temperature=0.8) for i in range(3)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name='collaborative_trust_meeting')
        meeting.agents.extend([system] + cot_agents)
        
        # Initial instruction to agents
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please assess the task: {task} and discuss your reasoning with each other."
        ))
        
        # Each agent presents their reasoning and trust levels
        coalition_agents = []
        trust_scores = {}
        trust_explanations = {}
        for agent in cot_agents:
            output = agent.forward(response_format={
                "reasoning": "Your reasoning for forming a coalition or going solo.",
                "trust": "Your trust score for each other agent (0-1)",
                "trust_explanation": "Explain how you derived your trust score"
            })
            meeting.chats.append(self.Chat(
                agent=agent,
                content=output["reasoning"] + f" (Trust: {output['trust']}, Explanation: {output['trust_explanation']})"
            ))
            
            # Collect trust scores and explanations without manual processing
            trust_scores[agent.agent_name] = float(output["trust"])
            trust_explanations[agent.agent_name] = output["trust_explanation"]
            if trust_scores[agent.agent_name] >= 0.5:  # Only consider agents with trust >= 0.5
                coalition_agents.append(agent)
        
        # If no coalitions were formed, proceed with individual answers
        if not coalition_agents:
            answers = []
            for agent in cot_agents:
                final_output = agent.forward(response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C, or D."
                })
                answers.append(final_output["answer"])
            final_answer = max(set(answers), key=answers.count)
        else:
            # Coalition formed, agents collaborate
            coalition_answers = []
            for agent in coalition_agents:
                final_output = agent.forward(response_format={
                    "thinking": "Your collaborative reasoning.",
                    "answer": "A single letter, A, B, C, or D."
                })
                coalition_answers.append((final_output["answer"], trust_scores[agent.agent_name]))
            
            # Weighted voting based on trust
            weighted_votes = {}
            for answer, trust in coalition_answers:
                if answer not in weighted_votes:
                    weighted_votes[answer] = 0
                weighted_votes[answer] += trust
            final_answer = max(weighted_votes, key=weighted_votes.get)
        
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
