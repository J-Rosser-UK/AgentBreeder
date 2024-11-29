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
        # Create a system agent to provide instructions
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create predator agents to evaluate outputs
        predator_agents = [
            self.Agent(agent_name=f'Predator Agent {i}', temperature=0.9)
            for i in range(2)
        ]
        
        # Create adversarial agents designed to exploit weaknesses
        adversarial_agents = [
            self.Agent(agent_name=f'Adversarial Agent {i}', temperature=0.9)
            for i in range(2)
        ]
        
        # Create prey agents to generate diverse outputs
        prey_agents = [
            self.Agent(agent_name=f'Prey Agent {i}', temperature=0.5)
            for i in range(3)
        ]
        
        # Create critic agents to question decisions
        critic_agents = [
            self.Agent(agent_name=f'Critic Agent {i}', temperature=0.6)
            for i in range(2)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="iterative_adversarial_dynamics_with_structured_critics")
        meeting.agents.extend([system] + predator_agents + adversarial_agents + prey_agents + critic_agents)
        
        # Instruct prey agents to generate outputs
        answers = []
        for prey in prey_agents:
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Please think step by step and solve the task: {task}"
            ))
            
            output = prey.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C, or D."
            })
            
            meeting.chats.append(self.Chat(
                agent=prey,
                content=output["thinking"] + output["answer"]
            ))
            
            answers.append(output["answer"])
    
        # Critics evaluate outputs and provide structured feedback
        for critic in critic_agents:
            meeting.chats.append(self.Chat(
                agent=system,
                content="Evaluate the answers from prey agents and provide a score along with specific feedback for improvement."
            ))
            
            critic_feedback = critic.forward(response_format={
                "answers": answers,
                "feedback": "Your feedback on the answers."
            })
            
            meeting.chats.append(self.Chat(
                agent=critic,
                content=critic_feedback["feedback"]
            ))
    
        # Prompt prey agents to revise their answers based on structured critiques
        revised_answers = []
        for prey in prey_agents:
            meeting.chats.append(self.Chat(
                agent=system,
                content="Given the structured critiques, please revise your answer."
            ))
            
            revised_output = prey.forward(response_format={
                "thinking": "Your revised step by step thinking.",
                "answer": "A single letter, A, B, C, or D."
            })
            
            meeting.chats.append(self.Chat(
                agent=prey,
                content=revised_output["thinking"] + revised_output["answer"]
            ))
            
            revised_answers.append(revised_output["answer"])
    
        # Determine the final answer based on revised outputs
        final_best_answer = max(set(revised_answers), key=revised_answers.count)  # Majority voting
        return final_best_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
