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
        # Create system and agent instances
        system = self.Agent(agent_name='system', temperature=0.8)
        cot_agents = [self.Agent(agent_name=f'Chain-of-Thought Agent {i}', temperature=0.7) for i in range(2)]
        critic_agents = [self.Agent(agent_name=f'Critic Agent {i}', temperature=0.6) for i in range(2)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_strategy_adjustment")
        meeting.agents.extend([system] + cot_agents + critic_agents)
        
        N_max = 3  # Maximum number of attempts
        answers = []  # Store answers from the initial attempts
        
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
                    "answer": "A single letter, A, B, C or D."
                },
                correct_answer=correct_answer  # Pass correct_answer here
            )
            meeting.chats.append(
                self.Chat(
                    agent=agent,
                    content=output["thinking"]
                )
            )
            answers.append(output["answer"])  # Store the answer directly
        
        performance_metrics = {agent.agent_name: 0 for agent in cot_agents}  # Initialize performance metrics
        
        # Feedback loop
        for i in range(N_max):
            meeting.chats.append(
                self.Chat(
                    agent=system,
                    content="Please review the answers and provide feedback on both individual and group performance."
                )
            )
            critic_outputs = [critic_agent.forward(response_format={"feedback": "Your detailed feedback."}) for critic_agent in critic_agents]
            
            for j, critic_output in enumerate(critic_outputs):
                meeting.chats.append(
                    self.Chat(
                        agent=critic_agents[j],
                        content=critic_output["feedback"]
                    )
                )
                
            # Adjust strategies based on performance metrics
            for agent in cot_agents:
                # Example: Increase temperature for underperforming agents
                if performance_metrics[agent.agent_name] < 1:  # Assuming a threshold for performance
                    agent.temperature += 0.1  # Adjust strategy
                meeting.chats.append(
                    self.Chat(
                        agent=system,
                        content="Given the feedback, refine your reasoning and try again: {task}"
                    )
                )
                output = agent.forward(
                    response_format={
                        "thinking": "Your step by step thinking.",
                        "answer": "A single letter, A, B, C or D."
                    },
                    correct_answer=correct_answer  # Pass correct_answer here
                )
                meeting.chats.append(
                    self.Chat(
                        agent=agent,
                        content=output["thinking"]
                    )
                )
                performance_metrics[agent.agent_name] += 1 if output["answer"] == correct_answer else 0  # Update performance
        
        # Select the final answer based on consensus of the outputs
        from collections import Counter
        final_answer = Counter(answers).most_common(1)[0][0]  # Return the most common answer
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
