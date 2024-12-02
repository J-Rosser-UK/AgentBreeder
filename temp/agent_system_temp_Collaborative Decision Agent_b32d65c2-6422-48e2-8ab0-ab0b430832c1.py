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

    def forward(self, task: str) -> str:
        # Create a system agent to provide concise instructions
        system = self.Agent(agent_name='system', temperature=0.8)
    
        # Initialize debate agents with different roles
        debate_agents = [self.Agent(
            agent_name=name,
            temperature=0.8
        ) for name in ['Biology Expert', 'Physics Expert', 'Science Generalist']]
    
        # Instruction for final decision-making
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup a single meeting for the debate
        meeting = self.Meeting(meeting_name="debate")
    
        # Ensure all agents are part of the meeting
        [meeting.agents.append(agent) for agent in debate_agents]
        meeting.agents.append(system)
        meeting.agents.append(final_decision_agent)
    
        max_round = 2  # Maximum number of debate rounds
    
        # Perform debate rounds
        for r in range(max_round):
            for i in range(len(debate_agents)):
                if r == 0 and i == 0:
                    meeting.chats.append(self.Chat(agent=system, content=f"Solve this: {task}"))
                    output = debate_agents[i].forward(response_format={"thinking": "Your reasoning.", "response": "Your answer.", "answer": "A, B, C, or D."})
                else:
                    meeting.chats.append(self.Chat(agent=system, content=f"Consider previous answers and update: {task}"))
                    output = debate_agents[i].forward(response_format={"thinking": "Your reasoning.", "response": "Your answer.", "answer": "A, B, C, or D."})
    
                meeting.chats.append(self.Chat(agent=debate_agents[i], content=output["thinking"] + output["response"]))
    
        # Final decision based on all debate results
        meeting.chats.append(self.Chat(agent=system, content="Review the reasoning from all agents and provide a final answer."))
        final_decision_output = final_decision_agent.forward(response_format={"reasoning": [output["thinking"] for output in output], "final_answer": "A, B, C, or D."})
        
        return final_decision_output["final_answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
