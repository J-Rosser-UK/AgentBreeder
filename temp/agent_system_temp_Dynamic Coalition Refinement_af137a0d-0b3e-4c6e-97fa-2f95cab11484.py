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
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_coalition_refinement")
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
            coalitions = []  # Track coalitions formed in this generation
            
            # Allow agents to evaluate and form coalitions
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
                
                # Form coalitions based on feedback and reasoning quality
                if feedback["feedback"] == "CORRECT":
                    coalitions.append(agent)
    
            # Allow agents to dynamically shift alliances based on feedback
            for agent in agents:
                if agent not in coalitions:
                    # Allow agent to join a successful coalition
                    if coalitions:
                        chosen_alliance = np.random.choice(coalitions)
                        meeting.chats.append(
                            self.Chat(
                                agent=agent,
                                content=f"I would like to join the coalition with {chosen_alliance.agent_name}"
                            )
                        )
                        coalitions.append(agent)  # Add to coalition
    
            # Evaluate coalition effectiveness and allow agents to vote on staying or shifting
            for agent in agents:
                if agent in coalitions:
                    vote = system.forward(
                        response_format={
                            "vote": "Should this coalition continue? (yes/no)"
                        }
                    )
                    meeting.chats.append(
                        self.Chat(
                            agent=agent,
                            content=f"Voting on coalition status: {vote['vote']}"
                        )
                    )
                    if vote["vote"] == "no":
                        coalitions.remove(agent)  # Remove from coalition if vote is no
    
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
