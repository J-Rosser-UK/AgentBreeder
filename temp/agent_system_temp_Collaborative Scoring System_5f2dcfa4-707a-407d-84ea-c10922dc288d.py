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
        # Create system and expert agents
        system = self.Agent(agent_name='system', temperature=0.8)
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Initialize scores
        scores = {name: 0 for name in expert_agents.keys()}
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_scoring_meeting")
        meeting.agents.extend([system] + list(expert_agents.values()))
        
        # Route the task to all expert agents
        for name, expert in expert_agents.items():
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"{name.capitalize()} Expert, please think step by step and provide your answer for the task: {task}"
            ))
            
            # Get answer and reasoning from the expert
            output = expert.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C or D."
            })
            
            # Score the expert based on the quality of their response
            scores[name] += score_response(output)
            
            # Record the chat
            meeting.chats.append(self.Chat(
                agent=expert,
                content=output["thinking"] + " Answer: " + output["answer"]
            ))
        
        # Discussion phase for peer feedback
        for name, expert in expert_agents.items():
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"{name.capitalize()} Expert, please critique the answers given by other experts."
            ))
            feedback_output = expert.forward(response_format={
                "feedback": "Your feedback on other experts' answers."
            })
            meeting.chats.append(self.Chat(
                agent=expert,
                content=feedback_output["feedback"]
            ))
    
        # Determine the expert with the highest score
        best_expert = max(scores, key=scores.get)
        
        # Ask the best expert to finalize their answer
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the scores, {best_expert.capitalize()} Expert, please provide your final answer for the task: {task}"
        ))
        
        final_output = expert_agents[best_expert].forward(response_format={
            "thinking": "Your final reasoning based on the task after feedback.",
            "answer": "A single letter, A, B, C or D."
        })
        
        return final_output["answer"]
    
    # Function to score responses (placeholder)
    def score_response(output):
        # Implement scoring logic based on reasoning clarity and correctness
        return 1  # Placeholder score

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
