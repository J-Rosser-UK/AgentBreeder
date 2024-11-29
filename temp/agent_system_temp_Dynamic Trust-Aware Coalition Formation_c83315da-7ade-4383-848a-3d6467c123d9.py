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
        
        # Initialize trust scores for agents
        trust_scores = {
            'Physics Expert': 1.0,
            'Chemistry Expert': 1.0,
            'Biology Expert': 1.0,
            'Science Generalist': 1.0
        }
        
        # Setup meeting for coalition formation
        meeting = self.Meeting(meeting_name="dynamic_trust_aware_coalition_meeting")
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
                "confidence": "Your confidence level (0-1)."
            })
            coalition_output = {
                "proposal": proposal_output["proposal"],
                "reasoning": proposal_output["reasoning"],
                "confidence": proposal_output["confidence"],
                "trust_score": trust_scores[agent.agent_name.split()[0]],  # Use only the main name
                "agent": agent
            }
            coalition_proposals.append(coalition_output)
            meeting.chats.append(self.Chat(
                agent=agent,
                content=proposal_output["proposal"] + " Reasoning: " + proposal_output["reasoning"] + " Confidence: " + str(proposal_output["confidence"])
            ))
        
        # Evaluate coalition proposals and select the best coalition based on confidence and trust
        selected_coalition = max(coalition_proposals, key=lambda x: x["confidence"] * x["trust_score"])  # Select the proposal with highest confidence * trust score
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
            coalition_output.append((output["answer"], output["confidence"], trust_scores[agent.agent_name.split()[0]]))  # Use only the main name
            meeting.chats.append(self.Chat(
                agent=agent,
                content=output["thinking"] + " Answer: " + output["answer"] + " Confidence: " + str(output["confidence"]) + " Trust Score: " + str(trust_scores[agent.agent_name.split()[0]])
            ))
        
        # Update trust scores based on performance
        for agent in coalition_agents:
            # Assume we now receive feedback from each agent about the correctness of their answer
            feedback_output = agent.forward(response_format={
                "feedback": "Was the answer correct? (yes/no)",
                "confidence": "Your confidence level (0-1)."
            })
            if feedback_output["feedback"] == 'yes':
                trust_scores[agent.agent_name.split()[0]] += 0.1  # Increase trust
            else:
                trust_scores[agent.agent_name.split()[0]] -= 0.1  # Decrease trust
        
        # Aggregate answers using a weighted approach based on confidence and trust levels
        final_answer = max(coalition_output, key=lambda x: x[1] * x[2])[0]  # Select answer with highest confidence * trust score
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
