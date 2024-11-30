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
        meeting = self.Meeting(meeting_name="collaborative_peer_review")
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
    
            # Peer-review process
            peer_feedbacks = []
            for i, agent in enumerate(agents):
                feedback = []
                for j, peer in enumerate(agents):
                    if i != j:
                        peer_feedback = peer.forward(
                            response_format={
                                "evaluation": f"Evaluate the answer from Agent {i}: {outputs[i][\'answer\']}"
                            }
                        )
                        if "evaluation" in peer_feedback:
                            feedback.append(peer_feedback["evaluation"])
                peer_feedbacks.append(feedback)
                meeting.chats.append(
                    self.Chat(
                        agent=agent, 
                        content=f"Peer feedback for Agent {i}: {feedback}"
                    )
                )
    
            # Dynamic role allocation based on peer feedback
            role_assignments = {"Critic": [], "Expert": [], "Generalist": []}
            for agent, feedback in zip(agents, peer_feedbacks):
                if feedback:
                    if any("CORRECT" in f for f in feedback):
                        role_assignments["Expert"].append(agent)
                    else:
                        role_assignments["Critic"].append(agent)
                else:
                    role_assignments["Generalist"].append(agent)
    
            # Assign remaining agents as Generalists
            role_assignments["Generalist"].extend([agent for agent in agents if agent not in role_assignments["Expert"] and agent not in role_assignments["Critic"]])
    
            # Strategy sharing and new generation creation
            successful_strategies = [output for output in outputs if "CORRECT" in output["thinking"]]
            if successful_strategies:
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
        final_answers = [output["answer"] for output in outputs]
        return max(set(final_answers), key=final_answers.count) if final_answers else "No answer"

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
