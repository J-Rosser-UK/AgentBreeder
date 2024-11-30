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
        meeting = self.Meeting(meeting_name="dynamic_trust_evaluation")
        meeting.agents.extend([system] + predator_agents + prey_agents + adversarial_agents)
        
        answers = []  # Initialize the answers list
        trust_scores = {prey.agent_id: 1.0 for prey in prey_agents}  # Initial trust scores for prey agents
        
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
            
            answers.append((output["answer"], trust_scores[prey.agent_id]))  # Append answer with trust score
    
        # Predator agents evaluate outputs and provide feedback
        for predator in predator_agents:
            meeting.chats.append(self.Chat(
                agent=system,
                content="Evaluate these answers: {answers} and provide feedback."
            ))
            
            predator_output = predator.forward(response_format={
                "answers": answers,
                "best_answer": "A single letter, A, B, C, or D.",
                "trust_adjustment": 0.1  # Default trust adjustment value
            })
            
            meeting.chats.append(self.Chat(
                agent=predator,
                content=f"Best answer: {predator_output['best_answer']}. Consider this next time."
            ))
    
            # Update trust scores based on feedback
            for answer, (prey_answer, prey_trust) in zip(predator_output['answers'], answers):
                if answer == predator_output['best_answer']:
                    trust_scores[prey.agent_id] += predator_output['trust_adjustment']  # Increase trust for correct answers
                else:
                    trust_scores[prey.agent_id] -= predator_output['trust_adjustment']  # Decrease trust for incorrect answers
    
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
    
            # Update trust scores based on critiques
            for prey in prey_agents:
                if adversarial_output['critique']:
                    trust_scores[prey.agent_id] -= 0.1  # Decrease trust for critiques
    
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
    
        # Return the best answer based on the critiques
        return predator_output['best_answer']

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
