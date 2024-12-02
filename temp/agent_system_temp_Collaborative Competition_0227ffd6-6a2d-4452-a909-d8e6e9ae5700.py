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
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_competition_meeting")
        meeting.agents.extend([system] + list(expert_agents.values()))
        
        # Each expert agent presents their solution
        scores = {}  # To keep track of scores for each expert
        for expert_name, expert in expert_agents.items():
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"{expert_name}, please think step by step and provide your best answer to: {task}"
            ))
            expert_output = expert.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            })
            scores[expert_name] = evaluate_answer(expert_output)  # Evaluate the answer and assign a score
            meeting.chats.append(self.Chat(
                agent=expert,
                content=expert_output["thinking"] + expert_output["answer"]
            ))
        
        # Provide feedback to each agent
        for expert_name in expert_agents.keys():
            feedback = generate_feedback(scores, expert_name)
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Feedback for {expert_name}: {feedback}"
            ))
            
        # Allow agents to revise their answers based on feedback
        for expert_name, expert in expert_agents.items():
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"{expert_name}, based on the feedback, please revise your answer to: {task}"
            ))
            expert_output = expert.forward(response_format={
                "thinking": "Your revised step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            })
            meeting.chats.append(self.Chat(
                agent=expert,
                content=expert_output["thinking"] + expert_output["answer"]
            ))
        
        # Determine the best answer based on updated scores
        best_expert = max(scores, key=scores.get)
        final_answer = expert_agents[best_expert].forward(response_format={
            "thinking": "Your final reasoning based on the best solution.",
            "answer": "A single letter, A, B, C or D."
        })
        return final_answer["answer"]
    
    
    def evaluate_answer(output):
        # Logic to evaluate the answer based on reasoning quality and correctness
        # Placeholder example: return 1 for correct reasoning, 0 for incorrect
        if output["answer"] in ["A", "B", "C", "D"]:
            return 1  # Assume correct for simplicity
        return 0  # Assume incorrect
    
    
    def generate_feedback(scores, expert_name):
        # Logic to generate feedback based on scores
        feedback = f"Your score is {scores[expert_name]}."
        if scores[expert_name] == 0:
            feedback += " Consider revising your reasoning."
        return feedback  # Return feedback for the expert

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
