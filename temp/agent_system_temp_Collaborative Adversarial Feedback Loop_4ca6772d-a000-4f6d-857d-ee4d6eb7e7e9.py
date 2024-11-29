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

    from collections import Counter
    
    def forward(self, task: str) -> str:
        # Create system and agent instances
        system = self.Agent(agent_name='system', temperature=0.8)
        population_size = 5  # Number of agents in each generation
        generations = 3  # Number of generations to evolve
        main_agents = [self.Agent(agent_name=f'Main Agent {i}', temperature=0.7) for i in range(population_size)]
        adversarial_agents = [self.Agent(agent_name=f'Adversarial Agent {i}', temperature=0.7) for i in range(population_size)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_adversarial_feedback_loop")
        meeting.agents.extend([system] + main_agents + adversarial_agents)
        
        for generation in range(generations):
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Generation {generation + 1}: Please solve the task: {task}"
                )
            )
            outputs = []
            for agent in main_agents:
                output = agent.forward(
                    response_format={
                        "thinking": "Your step by step thinking.",
                        "answer": "A single letter, A, B, C or D."
                    }
                )
                outputs.append(output)
                meeting.chats.append(
                    self.Chat(
                        agent=agent, 
                        content=output["thinking"]
                    )
                )
    
            # Adversarial critique
            critiques = {i: [] for i in range(population_size)}
            for adv_agent in adversarial_agents:
                for i, output in enumerate(outputs):
                    critique = adv_agent.forward(
                        response_format={
                            "critique": "Provide a counterargument or critique of the answer."
                        }
                    )
                    critiques[i].append(critique["critique"])
                    meeting.chats.append(
                        self.Chat(
                            agent=adv_agent,
                            content=f"{adv_agent.agent_name} critiques: {critique['critique']}"
                        )
                    )
    
            # Update main agents based on critiques
            revised_outputs = []
            for i, output in enumerate(outputs):
                feedback = "\n".join(critiques[i])
                meeting.chats.append(
                    self.Chat(
                        agent=system,
                        content=f"Feedback for {main_agents[i].agent_name}: {feedback}"
                    )
                )
                # Main agent revises based on feedback
                revised_output = main_agents[i].forward(
                    response_format={
                        "thinking": f"Given the feedback: {feedback}, revise your answer to the task: {task}, thinking step by step.",
                        "answer": "A single letter, A, B, C or D."
                    }
                )
                revised_outputs.append(revised_output)
    
            # Final consensus from the last generation using weighted voting
            final_answers = [output["answer"] for output in revised_outputs]
            answer_counts = Counter(final_answers)
            final_answer = max(answer_counts, key=answer_counts.get) if final_answers else "No answer"
            return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
