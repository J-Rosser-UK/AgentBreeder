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
        trust_agent = self.Agent(agent_name='Trust Evaluation Agent', temperature=0.7)
        lead_expert = self.Agent(agent_name='Lead Expert', temperature=0.9)
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="weighted_voting_meeting")
        meeting.agents.extend([system, trust_agent, lead_expert] + list(expert_agents.values()))
        
        # Each expert presents multiple reasoning outputs
        expert_outputs = []  # Store expert outputs for evaluation
        trust_scores = {}  # Store trust scores for each expert
        for expert in expert_agents.values():
            for i in range(3):  # Each expert generates 3 outputs
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f"{expert.agent_name}, please think step by step and present your reasoning for the task: {task}"
                ))
                expert_output = expert.forward(response_format={
                    "thinking": "Your step by step reasoning.",
                    "answer": "A single letter, A, B, C or D."
                })
                meeting.chats.append(self.Chat(
                    agent=expert,
                    content=expert_output.get("thinking", "No reasoning provided.")
                ))
                # Collect valid outputs
                if "answer" in expert_output:
                    expert_outputs.append(expert_output)  # Collect valid outputs
                    
                    # Evaluate trustworthiness for this expert's output
                    trust_score = trust_agent.forward(response_format={
                        "output": expert_output,
                        "trust_score": "The trust score based on the output."
                    })
                    trust_scores[expert.agent_name] = trust_score.get("trust_score", 0)
        
        # Voting process among experts with trust weights
        votes = {output['answer']: 0 for output in expert_outputs if 'answer' in output}
        total_weights = {output['answer']: 0 for output in expert_outputs if 'answer' in output}
        for output in expert_outputs:
            if 'answer' in output:
                answer = output['answer']
                votes[answer] += 1  # Count votes for each answer
                total_weights[answer] += trust_scores.get(output['agent_name'], 1)  # Weight votes by trust score
        
        # Lead Expert provides final recommendation
        meeting.chats.append(self.Chat(
            agent=lead_expert,
            content="Based on the reasoning provided, what is your final recommendation for the task?"
        ))
        lead_expert_output = lead_expert.forward(response_format={
            "recommendation": "Your recommended answer, A, B, C, or D."
        })
        
        # Determine final answer based on Lead Expert's recommendation or weighted votes
        if lead_expert_output and "recommendation" in lead_expert_output:
            final_answer = lead_expert_output["recommendation"]
        else:
            # Select answer based on weighted vote
            final_answer = max(total_weights, key=total_weights.get) if total_weights else "A"  # Default answer if no valid votes
        return final_answer  # Ensure the final answer is returned as a string.

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
