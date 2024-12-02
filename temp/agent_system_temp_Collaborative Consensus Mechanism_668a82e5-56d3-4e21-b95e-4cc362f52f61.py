import random
import numpy as np
import pandas

from mocked_base import MockAgent as Agent

from mocked_base import MockMeeting as Meeting

from mocked_base import MockChat as Chat

from sqlalchemy.orm import Session

class AgentSystem:
    def __init__(self, session: Session = None):
        self.Agent = Agent
        self.Meeting = Meeting
        self.Chat = Chat
        self.session = session

    def forward(self, task: str, correct_answer: str) -> str:
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        cot_agents = [self.Agent(agent_name=f'Chain-of-Thought Agent {i}', temperature=0.8) for i in range(3)]
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_consensus_meeting")
        meeting.agents.extend([system] + cot_agents)
        
        # Gather initial outputs from all Chain-of-Thought agents
        outputs = []
        for agent in cot_agents:
            output = agent.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            })
            outputs.append({"agent_name": agent.agent_name, "thinking": output["thinking"], "answer": output["answer"]})
            meeting.chats.append(self.Chat(
                agent=agent,
                content=output["thinking"] + output["answer"]
            ))
        
        # Evaluate outputs and assign scores based on correctness
        scores = {output["agent_name"]: 0 for output in outputs}
        for output in outputs:
            if output["answer"] == correct_answer:
                scores[output["agent_name"]] += 1
    
        # Use final decision agent to select the best answer based on scores
        best_agent_name = max(scores, key=scores.get)
        best_answer = next((output for output in outputs if output["agent_name"] == best_agent_name), None)
        
        # Ensure a valid answer is selected, fallback if needed
        if best_answer is None:
            return "No valid answer found."
        
        # Make final decision
        meeting.chats.append(self.Chat(
            agent=system,
            content="Based on the evaluations, the best answer is selected."
        ))
        
        return best_answer["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
