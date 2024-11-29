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
        debate_agents = [
            self.Agent(agent_name='Debate Agent 1', temperature=0.8),
            self.Agent(agent_name='Debate Agent 2', temperature=0.8)
        ]
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.5)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="weighted_debate_consensus_meeting")
        meeting.agents.extend([system] + debate_agents + [final_decision_agent])
        
        # System instruction to initiate the debate
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Let's discuss the task: {task}. Each agent will present their reasoning step by step."
        ))
        
        # Collect outputs from each debate agent
        for agent in debate_agents:
            output = agent.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C, or D.",
                "confidence": "Your confidence level (0-1)."
            })
            meeting.chats.append(self.Chat(
                agent=agent,
                content=output["thinking"] + f" Final answer: {output["answer"]} (Confidence: {output["confidence"]})"
            ))
        
        # Final decision based on the debate
        meeting.chats.append(self.Chat(
            agent=system,
            content="Based on the arguments presented, please provide a final consensus answer."
        ))
        
        # Final decision agent evaluates arguments and determines consensus answer
        final_output = final_decision_agent.forward(response_format={
            "evaluations": "Evaluate the arguments from the debate agents and provide a final answer.",
            "answer": "A single letter, A, B, C, or D."
        })
        
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
