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
        physics_agents = [self.Agent(agent_name=f'Physics Expert {i}', temperature=0.8) for i in range(1, 3)]
        chemistry_agents = [self.Agent(agent_name=f'Chemistry Expert {i}', temperature=0.8) for i in range(1, 3)]
        biology_agents = [self.Agent(agent_name=f'Biology Expert {i}', temperature=0.8) for i in range(1, 3)]
        generalist_agents = [self.Agent(agent_name=f'Science Generalist {i}', temperature=0.8) for i in range(1, 3)]
        
        # Setup meeting for discussion
        meeting = self.Meeting(meeting_name="collaborative_coalition_meeting")
        meeting.agents.extend([system] + physics_agents + chemistry_agents + biology_agents + generalist_agents)
        
        # Propose strategies based on task requirements
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the task: {task}, please propose your strategies for collaboration."
        ))
        
        # Collect proposals from each agent
        proposals = []
        for agent in meeting.agents[1:]:  # Skip the system agent
            proposal_output = agent.forward(response_format={
                "proposal": "Your strategy proposal.",
                "reasoning": "Your reasoning for this strategy.",
                "confidence": "Your confidence level (0-1)."
            })
            proposals.append(proposal_output)
            meeting.chats.append(self.Chat(
                agent=agent,
                content=proposal_output["proposal"] + " Reasoning: " + proposal_output["reasoning"] + " Confidence: " + str(proposal_output["confidence"])
            ))
        
        # Discussion phase for refining strategies
        meeting.chats.append(self.Chat(
            agent=system,
            content="Now, please discuss your proposals and refine them collectively."
        ))
        
        # Allow agents to refine their proposals based on discussion
        refined_proposals = []
        for agent in meeting.agents[1:]:  # Skip the system agent
            refined_output = agent.forward(response_format={
                "refined_proposal": "Your refined strategy proposal.",
                "reasoning": "Your reasoning for the refinement.",
                "confidence": "Your confidence level (0-1)."
            })
            refined_proposals.append(refined_output)
            meeting.chats.append(self.Chat(
                agent=agent,
                content=refined_output["refined_proposal"] + " Reasoning: " + refined_output["reasoning"] + " Confidence: " + str(refined_output["confidence"])
            ))
        
        # Select the best coalition based on refined proposals
        selected_coalition = max(refined_proposals, key=lambda x: x["confidence"])  # Select the proposal with the highest confidence
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
            coalition_output.append((output["answer"], output["confidence"]))  # Store answer with confidence
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
