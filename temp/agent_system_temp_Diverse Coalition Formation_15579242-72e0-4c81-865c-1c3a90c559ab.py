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
        # Create agents with different expertise
        system = self.Agent(agent_name='system', temperature=0.8)
        physics_agent = self.Agent(agent_name='Physics Expert', temperature=0.8)
        chemistry_agent = self.Agent(agent_name='Chemistry Expert', temperature=0.8)
        biology_agent = self.Agent(agent_name='Biology Expert', temperature=0.8)
        generalist_agent = self.Agent(agent_name='Science Generalist', temperature=0.8)
        
        # Setup meeting for coalition formation
        meeting = self.Meeting(meeting_name="diverse_coalition_meeting")
        meeting.agents.extend([system, physics_agent, chemistry_agent, biology_agent, generalist_agent])
        
        # Propose coalitions based on task requirements
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the task: {task}, please propose your coalitions and how you will collaborate to solve this task."
        ))
        
        # Collect coalition proposals from each agent
        coalition_proposals = []
        for agent in meeting.agents[1:]:  # Skip the system agent
            proposal_output = agent.forward(response_format={
                "proposal": "Your coalition proposal and strategy.",
                "reasoning": "Your reasoning for forming this coalition.",
                "confidence": "Your confidence level (0-1).",
                "diversity_score": "Your diversity score (0-1)."
            })
            try:
                coalition_output = {
                    "proposal": proposal_output["proposal"],
                    "reasoning": proposal_output["reasoning"],
                    "confidence": float(proposal_output["confidence"]),  # Ensure conversion to float
                    "diversity_score": float(proposal_output["diversity_score"])  # Ensure conversion to float
                }
            except ValueError:
                # Handle conversion errors by skipping this proposal or setting defaults
                continue
            coalition_proposals.append(coalition_output)
            meeting.chats.append(self.Chat(
                agent=agent,
                content=proposal_output["proposal"] + " Reasoning: " + proposal_output["reasoning"] + " Confidence: " + str(proposal_output["confidence"]) + " Diversity Score: " + str(proposal_output["diversity_score"])
            ))
        
        # Select coalition proposal based on confidence and diversity
        if not coalition_proposals:
            return "No valid coalition proposals available."
        total_weight = sum(p["confidence"] + p["diversity_score"] for p in coalition_proposals)
        selection_probs = [(p["confidence"] + p["diversity_score"]) / total_weight for p in coalition_proposals]  # Normalize
        selected_index = np.random.choice(len(coalition_proposals), p=selection_probs)  # Stochastic selection
        selected_coalition = coalition_proposals[selected_index]
        coalition_agents = [selected_coalition["agent"]]  # Extracting agent from selected proposal
        
        # Each coalition will work on the task
        meeting.chats.append(self.Chat(
            agent=system,
            content="Now, please collaborate within your coalition to solve the task step by step."
        ))
        coalition_output = []
        for agent in coalition_agents:
            output = agent.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C or D.",
                "confidence": "Your confidence level (0-1)."
            })
            coalition_output.append((output["answer"], float(output["confidence"])))  # Store answer with confidence
            meeting.chats.append(self.Chat(
                agent=agent,
                content=output["thinking"] + " Answer: " + output["answer"] + " Confidence: " + str(output["confidence"])
            ))
        
        # Aggregate answers using a weighted approach based on confidence levels
        final_answer = max(coalition_output, key=lambda x: x[1])[0]  # Select answer with highest confidence
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
