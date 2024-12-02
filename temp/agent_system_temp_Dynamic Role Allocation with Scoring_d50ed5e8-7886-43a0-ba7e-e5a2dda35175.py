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
        routing_agent = self.Agent(agent_name='Routing Agent', temperature=0.8)
        cot_agents = [self.Agent(agent_name=f'Chain-of-Thought Agent {i}', temperature=0.8) for i in range(3)]
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Initialize agent scores
        agent_scores = {agent.agent_name: 0 for agent in cot_agents}
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_role_allocation_meeting")
        meeting.agents.extend([system, routing_agent] + cot_agents)
        
        N_max = 3  # Maximum number of attempts
        
        # Initial attempt
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please think step by step and then solve the task: {task}"
        ))
        
        # Get initial outputs from all Chain-of-Thought agents
        outputs = []
        for agent in cot_agents:
            output = agent.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            })
            outputs.append(output)
            meeting.chats.append(self.Chat(
                agent=agent,
                content=output["thinking"] + output["answer"]
            ))
        
        # Evaluate outputs and update scores based on correctness
        for output in outputs:
            if output["answer"] == correct_answer:
                agent_scores[output["agent_name"]] += 1
    
        # Use routing agent to evaluate and select the best performing agent
        meeting.chats.append(self.Chat(
            agent=routing_agent,
            content="Based on the previous attempts, please choose the best agent to solve this task."
        ))
        routing_output = routing_agent.forward(response_format={
            "best_agent": "The name of the best agent"
        })
        best_agent_name = routing_output["best_agent"]
        best_agent = next((agent for agent in cot_agents if agent.agent_name == best_agent_name), None)
        
        # Ensure a valid agent is selected
        if best_agent is None:
            best_agent = cot_agents[0]  # Fallback to the first agent if none is valid
        
        # Make final decision using the best agent
        meeting.chats.append(self.Chat(
            agent=system,
            content="Given all the above solutions, reason over them carefully and provide a final answer."
        ))
        final_output = best_agent.forward(response_format={
            "thinking": "Your step by step thinking comparing all solutions.",
            "answer": "A single letter, A, B, C or D."
        })
        
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
