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
        coordinator = self.Agent(agent_name='Coordinator Agent', temperature=0.7)
        reasoning_agent = self.Agent(agent_name='Reasoning Agent', temperature=0.8)
        critique_agent = self.Agent(agent_name='Critique Agent', temperature=0.6)
        suggestion_agent = self.Agent(agent_name='Suggestion Agent', temperature=0.8)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="iterative_hierarchical_meeting")
        meeting.agents.extend([system, coordinator, reasoning_agent, critique_agent, suggestion_agent])
        
        # Coordinator instructs lower-level agents
        meeting.chats.append(self.Chat(
            agent=coordinator,
            content=f"Task: {task}. Reasoning Agent, please think step by step and provide your reasoning."
        ))
        reasoning_output = reasoning_agent.forward(response_format={
            "thinking": "Your step by step reasoning.",
            "answer": "A, B, C, or D."
        })
        meeting.chats.append(self.Chat(
            agent=reasoning_agent,
            content=reasoning_output["thinking"] + reasoning_output["answer"]
        ))
        
        # Coordinator asks Critique Agent to critique the reasoning
        meeting.chats.append(self.Chat(
            agent=coordinator,
            content=f"Critique the reasoning provided: {reasoning_output['thinking']} and answer: {reasoning_output['answer']}"
        ))
        critique_output = critique_agent.forward(response_format={
            "critique": "Your critique of the previous reasoning and answer."
        })
        meeting.chats.append(self.Chat(
            agent=critique_agent,
            content=critique_output["critique"]
        ))
        
        # Reasoning Agent refines its answer based on critique
        meeting.chats.append(self.Chat(
            agent=reasoning_agent,
            content=f"Based on the critique: {critique_output['critique']}, please refine your reasoning and answer."
        ))
        refined_output = reasoning_agent.forward(response_format={
            "thinking": "Your refined step by step reasoning.",
            "answer": "A, B, C, or D."
        })
        meeting.chats.append(self.Chat(
            agent=reasoning_agent,
            content=refined_output["thinking"] + refined_output["answer"]
        ))
        
        # Coordinator asks Suggestion Agent for alternative approaches
        for _ in range(3):  # Generate multiple suggestions
            meeting.chats.append(self.Chat(
                agent=coordinator,
                content=f"Suggest a new approach for: {task}."
            ))
            suggestion_output = suggestion_agent.forward(response_format={
                "thinking": "Your new reasoning.",
                "answer": "A, B, C, or D."
            })
            meeting.chats.append(self.Chat(
                agent=suggestion_agent,
                content=suggestion_output["thinking"] + suggestion_output["answer"]
            ))
        
        # Coordinator makes final decision based on all inputs
        meeting.chats.append(self.Chat(
            agent=coordinator,
            content="Review all solutions and critiques, and provide a final answer."
        ))
        final_output = coordinator.forward(response_format={
            "thinking": "Your reasoning based on all inputs.",
            "answer": "A, B, C, or D."
        })
        
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
