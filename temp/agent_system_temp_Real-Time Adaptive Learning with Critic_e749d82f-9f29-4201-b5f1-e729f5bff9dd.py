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
        critic = self.Agent(agent_name='critic', temperature=0.7)
        population_size = 5  # Number of agents
        agents = [self.Agent(agent_name=f'Agent {i}', temperature=0.7) for i in range(population_size)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="real_time_adaptive_learning_with_critic")
        meeting.agents.extend([system, critic] + agents)
        
        iterations = 3  # Number of iterations to adapt strategies
        
        for iteration in range(iterations):
            meeting.chats.append(
                self.Chat(
                    agent=system,
                    content=f"Iteration {iteration + 1}: Please solve the task: {task}"
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
                # Collect feedback from the critic agent
                feedback = critic.forward(
                    response_format={
                        "feedback": "Your evaluation of the answer."
                    }
                )
                meeting.chats.append(
                    self.Chat(
                        agent=critic,
                        content=feedback["feedback"]
                    )
                )
                # Adapt response based on feedback
                if feedback["feedback"] != "CORRECT":
                    agent.forward(response_format={
                        "thinking": "Adjust your response based on feedback."
                    })
    
        # Final consensus from the last iteration
        final_answers = [output["answer"] for output in outputs]
        return max(set(final_answers), key=final_answers.count) if final_answers else "No answer"

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
