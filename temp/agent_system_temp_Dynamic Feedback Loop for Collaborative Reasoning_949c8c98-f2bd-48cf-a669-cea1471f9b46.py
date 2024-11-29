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
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_feedback_meeting")
        meeting.agents.extend([system] + list(expert_agents.values()))
        
        expert_outputs = []  # Store expert outputs for evaluation
        for expert in expert_agents.values():
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"{expert.agent_name}, please think step by step and present your reasoning for the task: {task}"
            ))
            expert_output = expert.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C or D.",
                "confidence": "Your confidence score for the provided answer."
            })
            meeting.chats.append(self.Chat(
                agent=expert,
                content=expert_output["thinking"]
            ))
            expert_outputs.append(expert_output)  # Collect outputs
    
            # Feedback loop for peer evaluations
            for other_expert in expert_agents.values():
                if other_expert != expert:
                    meeting.chats.append(self.Chat(
                        agent=other_expert,
                        content=f"{expert.agent_name}'s reasoning was: {expert_output['thinking']}. What do you think?"
                    ))
                    feedback_output = other_expert.forward(response_format={
                        "feedback": "Your feedback on the reasoning.",
                        "confidence": "Your confidence in this feedback."
                    })
                    # Use feedback to refine the expert's answer
                    meeting.chats.append(self.Chat(
                        agent=expert,
                        content=f"Feedback received: {feedback_output['feedback']}. Please refine your reasoning."
                    ))
                    refined_output = expert.forward(response_format={
                        "thinking": "Your refined reasoning.",
                        "answer": "A single letter, A, B, C or D.",
                        "confidence": "Your confidence score for the refined answer."
                    })
                    expert_outputs[-1] = refined_output  # Update expert output with refined answer
    
        # Final decision based on refined outputs
        final_answer = max(expert_outputs, key=lambda x: x['confidence'])['answer']
        return final_answer  # Return the final answer based on the highest confidence.

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
