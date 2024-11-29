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
        predator1 = self.Agent(agent_name='Predator Agent 1', temperature=0.7)
        predator2 = self.Agent(agent_name='Predator Agent 2', temperature=0.7)
        prey1 = self.Agent(agent_name='Prey Agent 1', temperature=0.6)
        prey2 = self.Agent(agent_name='Prey Agent 2', temperature=0.6)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="adaptive_predator_prey_dynamics")
        meeting.agents.extend([system, predator1, predator2, prey1, prey2])
        
        N_max = 3  # Maximum number of iterations
        scores = {"predator1": 0, "predator2": 0}  # Initialize scores for predators
        
        # Initial attempts
        for i in range(N_max):
            # Predators generate solutions
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Please think step by step and then solve the task: {task}"
                )
            )
            output1 = predator1.forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            output2 = predator2.forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            
            meeting.chats.append(
                self.Chat(
                    agent=predator1, 
                    content=output1["thinking"]
                )
            )
            meeting.chats.append(
                self.Chat(
                    agent=predator2, 
                    content=output2["thinking"]
                )
            )
            
            # Prey provide feedback
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content="Please review the answers above and provide feedback."
                )
            )
            feedback1 = prey1.forward(
                response_format={
                    "feedback": "Your detailed feedback."
                }
            )
            feedback2 = prey2.forward(
                response_format={
                    "feedback": "Your detailed feedback."
                }
            )
            
            meeting.chats.append(
                self.Chat(
                    agent=prey1, 
                    content=feedback1["feedback"]
                )
            )
            meeting.chats.append(
                self.Chat(
                    agent=prey2, 
                    content=feedback2["feedback"]
                )
            )
            
            # Update scores based on feedback
            if "correct" in feedback1["feedback"]:
                scores["predator1"] += 1
            if "correct" in feedback2["feedback"]:
                scores["predator2"] += 1
            
            # Refine based on feedback
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content="Given the feedback, refine your reasoning and try again: {task}"
                )
            )
            output1 = predator1.forward(
                response_format={
                    "thinking": "Your refined step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            output2 = predator2.forward(
                response_format={
                    "thinking": "Your refined step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
        
        # Select the final answer based on the best predator's score
        if scores["predator1"] > scores["predator2"]:
            return output1["answer"]
        elif scores["predator2"] > scores["predator1"]:
            return output2["answer"]
        else:
            return "Ambiguous"

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
