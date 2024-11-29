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

    class AdaptivePerformanceBasedRoleAllocation:
        def evaluate_agent_performance(self, output, critique):
            # Example implementation of performance evaluation
            correct_answer = 'C'  # Placeholder for the actual correct answer
            score = 0
            if output['answer'] == correct_answer:
                score += 1  # Increment score for correct answer
            # Further evaluation could include assessing the critique quality
            if critique and 'good' in critique['critique'].lower():
                score += 0.5  # Increment score for a good critique
            return score
        
        def forward(self, task: str) -> str:
            # Create system and agent instances
            system = self.Agent(agent_name='system', temperature=0.8)
            population_size = 5  # Number of agents in each generation
            generations = 3  # Number of generations to evolve
            main_agents = [self.Agent(agent_name=f'Main Agent {i}', temperature=0.7) for i in range(population_size)]
            adversarial_agents = [self.Agent(agent_name=f'Adversarial Agent {i}', temperature=0.7) for i in range(population_size)]
            
            # Setup meeting
            meeting = self.Meeting(meeting_name="adaptive_role_allocation")
            meeting.agents.extend([system] + main_agents + adversarial_agents)
            
            for generation in range(generations):
                meeting.chats.append(
                    self.Chat(
                        agent=system, 
                        content=f"Generation {generation + 1}: Please solve the task: {task}"
                    )
                )
                outputs = []
                critiques = []
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
                for adv_agent in adversarial_agents:
                    for i, output in enumerate(outputs):
                        critique = adv_agent.forward(
                            response_format={
                                "critique": "Provide a counterargument or critique of the answer."
                            }
                        )
                        critiques.append((i, critique))  # Store index and critique together
                        meeting.chats.append(
                            self.Chat(
                                agent=adv_agent,
                                content=critique["critique"]
                            )
                        )
    
                # Evaluate performance of main agents
                performance_scores = []
                for i, output in enumerate(outputs):
                    score = self.evaluate_agent_performance(output, critiques[i][1])  # Use critique directly
                    performance_scores.append(score)
    
                # Dynamic role allocation based on performance
                sorted_agents = sorted(zip(main_agents, performance_scores), key=lambda x: x[1], reverse=True)
                new_main_agents = [agent for agent, score in sorted_agents[:population_size // 2]]
                new_adversarial_agents = [agent for agent, score in sorted_agents[population_size // 2:]]
    
                # Update meeting agents
                meeting.agents = [system] + new_main_agents + new_adversarial_agents
    
            # Final consensus from the last generation
            final_answers = [output["answer"] for output in outputs]
            return max(set(final_answers), key=final_answers.count) if final_answers else "No answer"

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
