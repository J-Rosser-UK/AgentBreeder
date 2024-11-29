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
        # Create system and agent instances
        system = self.Agent(agent_name='system', temperature=0.8)
        population_size = 10  # Number of agents in each generation
        generations = 3  # Number of generations to evolve
        agents = [self.Agent(agent_name=f'Agent {i}', temperature=0.7) for i in range(population_size)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_evolution")
        meeting.agents.extend([system] + agents)
        
        for generation in range(generations):
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Generation {generation + 1}: Please solve the task: {task}"
                )
            )
            outputs = []
            for agent in agents:
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
    
            # Engage in collaborative discussions
            for i in range(len(agents)):
                for j in range(i + 1, len(agents)):
                    discussion_output = self.Agent(agent_name='Discussion Agent', temperature=0.6).forward(
                        response_format={
                            "discussion": f"Discuss the reasoning between Agent {i} and Agent {j}."
                        }
                    )
                    meeting.chats.append(
                        self.Chat(
                            agent=system,
                            content=discussion_output["discussion"]
                        )
                    )
    
            # Collect feedback for each agent's output
            feedbacks = []
            for output in outputs:
                feedback = system.forward(
                    response_format={
                        "feedback": "Your evaluation of the answer."
                    }
                )
                feedbacks.append(feedback)
                meeting.chats.append(
                    self.Chat(
                        agent=system,
                        content=feedback["feedback"]
                    )
                )
    
            # Strategy sharing and new generation creation
            successful_strategies = [output for output, feedback in zip(outputs, feedbacks) if feedback["feedback"] == "CORRECT"]
            if successful_strategies:
                # Create new agents based on successful strategies
                new_agents = []
                for i in range(population_size):
                    strategy = np.random.choice(successful_strategies)
                    new_agent = self.Agent(agent_name=f'New Agent {i}', temperature=0.7)
                    new_agent.strategy = strategy["thinking"] + " with a collaborative twist!"
                    new_agents.append(new_agent)
                    meeting.agents.append(new_agent)
                agents = new_agents  # Update agents for the next generation
            else:
                break  # Stop if no successful strategies were found
    
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
