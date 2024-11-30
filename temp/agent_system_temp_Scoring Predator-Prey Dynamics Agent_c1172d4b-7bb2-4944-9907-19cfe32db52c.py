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
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        prey_agents = [
            self.Agent(agent_name=f'Prey Agent {i}', temperature=0.7) for i in range(2)
        ]
        predator_agents = [
            self.Agent(agent_name=f'Predator Agent {i}', temperature=0.8) for i in range(2)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name='predator_prey_meeting')
        meeting.agents.extend([system] + prey_agents + predator_agents)
        
        # Prey agents attempt to solve the task
        prey_outputs = []
        for prey in prey_agents:
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Please think step by step and solve the task: {task}"
            ))
            prey_output = prey.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C, or D."
            })
            prey_outputs.append(prey_output)  # Store the entire output object
            meeting.chats.append(self.Chat(
                agent=prey,
                content=prey_output["thinking"] + prey_output["answer"]
            ))
        
        # Predator agents critique the prey solutions
        for predator in predator_agents:
            meeting.chats.append(self.Chat(
                agent=system,
                content="Review the solutions provided by the prey agents and provide detailed feedback and scores."
            ))
            for output in prey_outputs:
                feedback_output = predator.forward(response_format={
                    "feedback": "Your critique of the solution.",
                    "score": "Score from 1 to 10 based on correctness and reasoning."
                })
                meeting.chats.append(self.Chat(
                    agent=predator,
                    content=feedback_output["feedback"] + f" Score: {feedback_output["score"]}"
                ))
        
        # Prey agents refine their answers based on predator feedback
        refined_solutions = []
        for i, prey in enumerate(prey_agents):
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Given the feedback and scores from predators, refine your solution: {prey_outputs[i]["answer"]}"
            ))
            refined_output = prey.forward(response_format={
                "thinking": "Your refined step by step thinking.",
                "answer": "A single letter, A, B, C, or D."
            })
            refined_solutions.append(refined_output["answer"])
            meeting.chats.append(self.Chat(
                agent=prey,
                content=refined_output["thinking"] + refined_output["answer"]
            ))
        
        # Aggregate refined answers to produce final output
        final_answer = max(set(refined_solutions), key=refined_solutions.count)  # Majority voting
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
