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
        principle_agents = [self.Agent(agent_name=f'Principle Agent {i}', temperature=0.8) for i in range(2)]
        cot_agents = [self.Agent(agent_name=f'Chain-of-Thought Agent {i}', temperature=0.8) for i in range(2)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="sophisticated_coalition_meeting")
        meeting.agents.extend([system] + principle_agents + cot_agents)
        
        # Propose alliances based on strengths
        for agent in principle_agents:
            meeting.chats.append(self.Chat(
                agent=agent,
                content="What are your strengths and confidence levels?"
            ))
        
        # Collect responses about strengths and confidence levels
        strengths_and_confidences = []
        for agent in principle_agents:
            output = agent.forward(response_format={
                "strengths": "Your strengths and how you can contribute.",
                "confidence": "Your confidence level (0-1)."
            })
            strengths_and_confidences.append((output["strengths"], output["confidence"], agent))
            meeting.chats.append(self.Chat(agent=agent, content=output["strengths"] + f' Confidence: {output["confidence"]}'))
        
        # Form coalitions based on confidence levels
        coalition = [agent for _, confidence, agent in strengths_and_confidences if confidence > 0.5]  # Example threshold
        
        # Ensure we have a coalition
        if not coalition:
            return "No coalition could be formed."
        
        # Solve the task using the coalition
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Coalition formed. Now, please solve the task: {task}"
        ))
        
        coalition_outputs = []
        for agent in coalition:
            output = agent.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            })
            coalition_outputs.append(output)
            meeting.chats.append(self.Chat(agent=agent, content=output["thinking"] + output["answer"]))
        
        # Aggregate answers from the coalition using a simple voting mechanism
        from collections import Counter
        final_answer = Counter([output["answer"] for output in coalition_outputs]).most_common(1)[0][0]
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
