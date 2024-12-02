import random
import numpy as np
import pandas

from mocked_base import MockAgent as Agent

from mocked_base import MockMeeting as Meeting

from mocked_base import MockChat as Chat

from sqlalchemy.orm import Session

class AgentSystem:
    def __init__(self, session: Session = None):
        self.Agent = Agent
        self.Meeting = Meeting
        self.Chat = Chat
        self.session = session

    def forward(self, task: str) -> str:
        # Create system and agent instances
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        cot_agent = self.Agent(
            agent_name='Chain-of-Thought Agent',
            temperature=0.7
        )
        
        critic_agent = self.Agent(
            agent_name='Critic Agent',
            temperature=0.6
        )
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="structured_feedback")
        meeting.agents.extend([system, cot_agent, critic_agent])
        
        N_max = 3  # Maximum number of attempts
        
        # Initial attempt
        meeting.chats.append(
            self.Chat(
                agent=system, 
                content=f"Please think step by step and then solve the task: {task}"
            )
        )
        
        output = cot_agent.forward(
            response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            }
        )
        
        meeting.chats.append(
            self.Chat(
                agent=cot_agent, 
                content=output["thinking"]
            )
        )
        
        # Refinement loop
        for i in range(N_max):
            # Introduce structured noise based on task complexity
            task_complexity = len(task.split())  # Example metric for complexity
            noise_factor = min(max(0.1, task_complexity / 10), 0.5)  # Adjust noise based on complexity
            noise_probs = [noise_factor, noise_factor, (1 - noise_factor) / 2, (1 - noise_factor) / 2]
            noise = np.random.choice(['A', 'B', 'C', 'D'], p=noise_probs)
            noisy_answer = output["answer"] if np.random.rand() > 0.2 else noise
            
            # Get feedback from critic
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Please review the answer above: {noisy_answer}. Critique where it might be wrong. If correct, output 'CORRECT'."
                )
            )
            
            critic_output = critic_agent.forward(
                response_format={
                    "feedback": "Your detailed feedback.",
                    "correct": "Either 'CORRECT' or 'INCORRECT'"
                }
            )
            
            meeting.chats.append(
                self.Chat(
                    agent=critic_agent, 
                    content=critic_output["feedback"]
                )
            )
            
            if critic_output["correct"] == "CORRECT":
                return output["answer"]  # Directly return the answer if correct
            
            # Reflect and refine
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Given the feedback above, consider where you could go wrong in your latest attempt. Using these insights, try to solve the task better: {task}"
                )
            )
            
            output = cot_agent.forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            
            meeting.chats.append(
                self.Chat(
                    agent=cot_agent, 
                    content=output["thinking"]
                )
            )
        
        return output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
