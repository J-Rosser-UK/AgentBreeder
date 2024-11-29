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

    def forward(self, task: str, correct_answer: str) -> str:
        import time
        # Create system and agent instances
        system = self.Agent(agent_name='system', temperature=0.8)
        cot_agents = [self.Agent(agent_name=f'Chain-of-Thought Agent {i}', temperature=0.7) for i in range(2)]
        critic_agents = [self.Agent(agent_name=f'Critic Agent {i}', temperature=0.6) for i in range(2)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_role_allocation")
        meeting.agents.extend([system] + cot_agents + critic_agents)
        
        N_max = 3  # Maximum number of attempts
        performance = {agent.agent_name: 0 for agent in cot_agents}  # Track performance of CoT agents
        
        # Initial attempts
        for agent in cot_agents:
            meeting.chats.append(
                self.Chat(
                    agent=system,
                    content=f"Please think step by step and then solve the task: {task}"
                )
            )
            output = agent.forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C, or D."
                }
            )
            meeting.chats.append(
                self.Chat(
                    agent=agent,
                    content=output["thinking"]
                )
            )
            performance[agent.agent_name] += 1 if output["answer"] == correct_answer else 0  # Check against correct answer
        
        # Refinement loop
        for i in range(N_max):
            # Determine the best performing agent
            best_agent_name = max(performance, key=performance.get)
            best_agent = next(agent for agent in cot_agents if agent.agent_name == best_agent_name)
            
            # Get feedback from critics
            meeting.chats.append(
                self.Chat(
                    agent=system,
                    content="Please review the answers above and provide feedback."
                )
            )
            for critic in critic_agents:
                critic_output = critic.forward(
                    response_format={
                        "feedback": "Your detailed feedback."
                    }
                )
                meeting.chats.append(
                    self.Chat(
                        agent=critic,
                        content=critic_output["feedback"]
                    )
                )
            
            # Reflect and refine based on feedback
            meeting.chats.append(
                self.Chat(
                    agent=system,
                    content=f"Given the feedback, refine your reasoning and try again: {task}"
                )
            )
            
            # Get output from the best performing agent
            output = best_agent.forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C, or D."
                }
            )
            meeting.chats.append(
                self.Chat(
                    agent=best_agent,
                    content=output["thinking"]
                )
            )
            performance[best_agent.agent_name] += 1 if output["answer"] == correct_answer else 0  # Update performance metric
        
        # Select the final answer based on the best agent's output
        return output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
