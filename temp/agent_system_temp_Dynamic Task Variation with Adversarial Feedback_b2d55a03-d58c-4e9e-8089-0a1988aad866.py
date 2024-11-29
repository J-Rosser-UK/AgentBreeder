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
        system = self.Agent(agent_name='system', temperature=0.8)
        
        # Create predator agents to evaluate outputs
        predator_agents = [
            self.Agent(agent_name=f'Predator Agent {i}', temperature=0.9)
            for i in range(2)
        ]
        
        # Create prey agents to generate diverse outputs
        prey_agents = [
            self.Agent(agent_name=f'Prey Agent {i}', temperature=0.5)
            for i in range(3)
        ]
        
        # Create adversarial agents to critique the outputs
        adversarial_agents = [
            self.Agent(agent_name=f'Adversarial Agent {i}', temperature=0.9)
            for i in range(2)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_task_variation")
        meeting.agents.extend([system] + predator_agents + prey_agents + adversarial_agents)
        
        answers = []  # Initialize the answers list
        
        # Instruct prey agents to generate outputs
        for prey in prey_agents:
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Solve this task: {task}"
            ))
            
            output = prey.forward(response_format={
                "thinking": "Your reasoning.",
                "answer": "A single letter, A, B, C, or D."
            })
            
            meeting.chats.append(self.Chat(
                agent=prey,
                content=output["thinking"] + output["answer"]
            ))
            
            answers.append(output["answer"])
    
        # Predator agents evaluate outputs and provide feedback
        for predator in predator_agents:
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Evaluate these answers: {answers} and provide feedback."
            ))
            
            predator_output = predator.forward(response_format={
                "answers": answers,
                "best_answer": "A single letter, A, B, C, or D."
            })
            
            meeting.chats.append(self.Chat(
                agent=predator,
                content=f"Best answer evaluated: {predator_output['best_answer']}."
            ))
    
        # Allow adversarial agents to critique the outputs
        critiques = []
        for adversary in adversarial_agents:
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Critique the outputs and feedback: {answers} and {predator_output['best_answer']}. What weaknesses do you see?"
            ))
            adversarial_output = adversary.forward(response_format={
                "critique": "Your critique of the answers.",
                "suggestions": "Suggestions for improvement."
            })
            meeting.chats.append(self.Chat(
                agent=adversary,
                content=adversarial_output["critique"] + adversarial_output["suggestions"]
            ))
            critiques.append(adversarial_output)
    
        # Refine prey agents' responses based on consolidated feedback
        for prey in prey_agents:
            meeting.chats.append(self.Chat(
                agent=system,
                content="Based on the critiques, refine your answer."
            ))
            refined_output = prey.forward(response_format={
                "thinking": "Your refined reasoning.",
                "answer": "A single letter, A, B, C, or D."
            })
            meeting.chats.append(self.Chat(
                agent=prey,
                content=refined_output["thinking"] + refined_output["answer"]
            ))
    
        # Introduce a task variation mechanism
        new_task = f"Modified version of the task based on feedback: {task}"
        # Handle task variation without recursion
        return self.forward(new_task)

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
