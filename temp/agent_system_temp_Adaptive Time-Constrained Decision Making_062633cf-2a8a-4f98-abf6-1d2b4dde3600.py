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
        import time
        
        # Helper function for time-constrained execution
        def time_constrained_execution(agent, response_format, time_limit=5):
            start_time = time.time()
            output = agent.forward(response_format=response_format)
            elapsed_time = time.time() - start_time
            if elapsed_time > time_limit:
                return {'thinking': 'Default reasoning due to time limit.', 'answer': 'C'}  # Default answer if time exceeded
            if not output or 'answer' not in output:
                return {'thinking': 'Fallback due to missing output.', 'answer': 'C'}  # Fallback if output is not valid
            return output
        
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        routing_agent = self.Agent(agent_name='Routing Agent', temperature=0.8)
        critic_agent = self.Agent(agent_name='Critic Agent', temperature=0.7)
        
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name='role_assignment_meeting')
        meeting.agents.extend([system, routing_agent, critic_agent] + list(expert_agents.values()))
        
        # Route the task
        meeting.chats.append(self.Chat(
            agent=system,
            content='Given the task, please choose an Expert to answer the question. Choose from: Physics, Chemistry, Biology Expert, or Science Generalist.'
        ))
        
        routing_output = time_constrained_execution(routing_agent, response_format={
            'choice': 'One of: physics, chemistry, biology, or general'
        }, time_limit=5)  # Adaptive time limit based on task complexity
        
        # Select expert based on routing decision
        expert_choice = routing_output['choice'].lower()
        if expert_choice not in expert_agents:
            expert_choice = 'general'
            
        selected_expert = expert_agents[expert_choice]
        
        # Get answer from selected expert
        meeting.chats.append(self.Chat(
            agent=system,
            content=f'Please think step by step and then solve the task: {task}'
        ))
        
        expert_output = time_constrained_execution(selected_expert, response_format={
            'thinking': 'Your step by step thinking.',
            'answer': 'A single letter, A, B, C or D.'
        }, time_limit=5)  # Adaptive time limit
        
        # Critic reviews the expert's answer
        meeting.chats.append(self.Chat(
            agent=system,
            content='Critic, please review the expert's answer and reasoning. Provide detailed feedback and suggestions.'
        ))
        
        critic_output = time_constrained_execution(critic_agent, response_format={
            'feedback': 'Your detailed feedback and suggestions.'
        }, time_limit=5)  # Adaptive time limit
        
        # Always ask the expert to consider the critic's feedback
        meeting.chats.append(self.Chat(
            agent=system,
            content='Given the critic's feedback, please revise your answer and reasoning: {task}'
        ))
        
        revised_expert_output = time_constrained_execution(selected_expert, response_format={
            'thinking': 'Your step by step thinking after feedback.',
            'answer': 'A single letter, A, B, C, or D.'
        }, time_limit=5)  # Adaptive time limit
        
        return revised_expert_output['answer']

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
