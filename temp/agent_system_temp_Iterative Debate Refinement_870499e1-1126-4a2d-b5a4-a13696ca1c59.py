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
        # Create a system agent to provide instructions
        system = self.Agent(agent_name='system', temperature=0.8)
    
        # Initialize debate agents with different roles and a moderate temperature for varied reasoning
        debate_agents = [self.Agent(
            agent_name=name,
            temperature=0.8
        ) for name in ['Biology Expert', 'Physics Expert', 'Science Generalist']]
    
        # Instruction for final decision-making based on all debates and solutions
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup a single meeting for the debate
        meeting = self.Meeting(meeting_name="debate")
    
        # Ensure all agents are part of the meeting
        [meeting.agents.append(agent) for agent in debate_agents]
        meeting.agents.append(system)
        meeting.agents.append(final_decision_agent)
    
        max_round = 2 # Maximum number of debate rounds
    
        responses = []  # Store responses for reflection
    
        # Perform debate rounds
        for r in range(max_round):
            for i in range(len(debate_agents)):
                if r == 0:
                    meeting.chats.append(self.Chat(agent=system, content=f"Solve the task: {task}"))
                    output = debate_agents[i].forward(response_format={"thinking": "Your reasoning.", "response": "Your answer.", "answer": "A, B, C, or D."})
                else:
                    previous_responses = '\n'.join(responses)  # Reflect on previous responses
                    meeting.chats.append(self.Chat(agent=system, content=f"Consider the previous responses:\n{previous_responses}\nNow, update your answer for the task: {task}"))
                    output = debate_agents[i].forward(response_format={"thinking": "Your reasoning.", "response": "Your answer.", "answer": "A, B, C, or D."})
    
                responses.append(output)  # Store the structured response directly
    
        # Make the final decision based on all debate results and solutions
        meeting.chats.append(self.Chat(agent=system, content="Summarize the answers and provide a final answer based on the reasoning depth."))
        output = final_decision_agent.forward(response_format={"thinking": "Your reasoning based on all responses.", "answer": "A, B, C, or D."})
        
        return output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
