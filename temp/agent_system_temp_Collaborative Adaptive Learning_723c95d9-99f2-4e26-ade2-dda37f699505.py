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
        population_size = 5  # Number of agents in each generation
        generations = 3  # Number of generations to evolve
        agents = [self.Agent(agent_name=f'Agent {i}', temperature=0.7) for i in range(population_size)]
        performance_history = {agent.agent_id: [] for agent in agents}  # Track performance history
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_adaptive_learning")
        meeting.agents.extend([system] + agents)
        
        for generation in range(generations):
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Generation {generation + 1}: Please solve the task: {task}"
                )
            )
            outputs = []
            feedbacks = []
            # Pair agents for collaborative problem solving
            for i in range(0, population_size, 2):
                agent1 = agents[i]
                agent2 = agents[i + 1] if i + 1 < population_size else agents[0]  # Pair with the first agent if odd number
                output1 = agent1.forward(
                    response_format={
                        "thinking": "Your step by step thinking.",
                        "answer": "A single letter, A, B, C or D."
                    }
                )
                output2 = agent2.forward(
                    response_format={
                        "thinking": "Your step by step thinking.",
                        "answer": "A single letter, A, B, C or D."
                    }
                )
                outputs.append((output1, output2))
                meeting.chats.append(
                    self.Chat(
                        agent=agent1, 
                        content=output1["thinking"]
                    )
                )
                meeting.chats.append(
                    self.Chat(
                        agent=agent2, 
                        content=output2["thinking"]
                    )
                )
                # Collect feedback from each other
                feedback1 = agent2.forward(
                    response_format={
                        "feedback": "Your evaluation of Agent 1's answer."
                    }
                )
                feedback2 = agent1.forward(
                    response_format={
                        "feedback": "Your evaluation of Agent 2's answer."
                    }
                )
                feedbacks.append((feedback1, feedback2))
    
            # Dynamic role allocation based on peer feedback
            role_assignments = {"Expert": [], "Generalist": []}
            for (output1, output2), (feedback1, feedback2) in zip(outputs, feedbacks):
                performance_history[output1["agent_id"]].append(feedback1["feedback"])
                performance_history[output2["agent_id"]].append(feedback2["feedback"])
                if feedback1["feedback"] == "CORRECT":
                    role_assignments["Expert"].append(agent1)
                else:
                    role_assignments["Generalist"].append(agent1)
                if feedback2["feedback"] == "CORRECT":
                    role_assignments["Expert"].append(agent2)
                else:
                    role_assignments["Generalist"].append(agent2)
    
            # Strategy sharing and new generation creation
            successful_strategies = [output for (output1, output2), (feedback1, feedback2) in zip(outputs, feedbacks) if feedback1["feedback"] == "CORRECT" or feedback2["feedback"] == "CORRECT"]
            if successful_strategies:
                # Create new agents based on successful strategies
                new_agents = []
                for i in range(population_size):
                    strategy = np.random.choice(successful_strategies)
                    new_agent = self.Agent(agent_name=f'New Agent {i}', temperature=0.7)
                    new_agent.strategy = strategy["thinking"] + " with learned adaptations!"
                    new_agents.append(new_agent)
                    meeting.agents.append(new_agent)
                agents = new_agents  # Update agents for the next generation
            else:
                break  # Stop if no successful strategies were found
    
        # Final consensus from the last generation
        final_answers = [output[0]["answer"] for output in outputs] + [output[1]["answer"] for output in outputs]
        # Use majority voting for final answer without manual aggregation
        return max(set(final_answers), key=final_answers.count) if final_answers else "No answer"

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
