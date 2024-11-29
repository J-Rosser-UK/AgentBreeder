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
        routing_agent = self.Agent(agent_name='Routing Agent', temperature=0.8)
        principle_agent = self.Agent(agent_name='Principle Agent', temperature=0.8)
        cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="adaptive_performance_learning_meeting")
        meeting.agents.extend([system, routing_agent, principle_agent, cot_agent])
        
        # Route the task to determine roles based on performance
        meeting.chats.append(self.Chat(
            agent=system,
            content="Given the task, evaluate the performance of each agent and assign roles accordingly."
        ))
        
        routing_output = routing_agent.forward(response_format={
            "role_assignments": "JSON object with agent roles and performance scores"
        })
        
        # Validate routing_output
        if isinstance(routing_output, str):
            try:
                routing_output = json.loads(routing_output)
            except json.JSONDecodeError:
                return "Error: Routing agent did not return valid role assignments."
        
        # Check if routing_output is valid and structured
        if not isinstance(routing_output, dict) or "role_assignments" not in routing_output:
            return "Error: Routing agent did not return valid role assignments."
        
        role_assignments = routing_output["role_assignments"]
        
        # Check if the principle agent is assigned
        if role_assignments.get('principle_agent'):
            meeting.chats.append(self.Chat(
                agent=system,
                content="What principles are involved in solving this task? Please think step by step."
            ))
            principle_output = principle_agent.forward(response_format={
                "thinking": "Your step by step reasoning about the principles.",
                "principles": "List and explanation of the principles involved."
            })
            meeting.chats.append(self.Chat(
                agent=principle_agent,
                content=principle_output["thinking"] + principle_output["principles"]
            ))
        
        # Now solve using the assigned roles
        if role_assignments.get('cot_agent'):
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Given the question and principles above, think step by step and solve the task: {task}"
            ))
            final_output = cot_agent.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C, or D."
            })
            return final_output["answer"]
        
        # Fallback for no valid agent assigned
        return "No valid agent assigned to solve the task. Please ensure agents are correctly assigned."

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
