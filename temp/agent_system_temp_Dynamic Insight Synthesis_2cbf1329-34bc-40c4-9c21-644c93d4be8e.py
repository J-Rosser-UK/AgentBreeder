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
        # Create a Lead Agent to guide the process
        lead_agent = self.Agent(
            agent_name='Lead Agent',
            temperature=0.9
        )
        
        # Create Supporting Agents with specialized knowledge
        supporting_agents = [
            self.Agent(agent_name='Supporting Agent 1', temperature=0.7),
            self.Agent(agent_name='Supporting Agent 2', temperature=0.7)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_insight_synthesis")
        meeting.agents.extend([lead_agent] + supporting_agents)
        
        # Lead Agent provides instructions
        meeting.chats.append(
            self.Chat(
                agent=lead_agent,
                content=f"Given the task: {task}, please provide your insights step by step."
            )
        )
        
        # Gather responses from Supporting Agents
        responses = []
        for agent in supporting_agents:
            output = agent.forward(
                response_format={
                    "thinking": "Your step by step reasoning.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            meeting.chats.append(
                self.Chat(
                    agent=agent,
                    content=output["thinking"]
                )
            )
            responses.append(output)
        
        # Lead Agent evaluates responses and asks clarifying questions if needed
        for output in responses:
            follow_up_content = "Can you clarify your reasoning further?"
            meeting.chats.append(
                self.Chat(
                    agent=lead_agent,
                    content=follow_up_content
                )
            )
            clarifying_output = output["agent"].forward(
                response_format={
                    "thinking": "Your refined reasoning.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            meeting.chats.append(
                self.Chat(
                    agent=output["agent"],
                    content=clarifying_output["thinking"]
                )
            )
        
        # Lead Agent synthesizes final answer
        meeting.chats.append(
            self.Chat(
                agent=lead_agent,
                content="Based on the insights provided, please determine the final answer."
            )
        )
        
        final_output = lead_agent.forward(
            response_format={
                "final_answer": "A single letter, A, B, C or D."
            }
        )
        
        return final_output["final_answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
