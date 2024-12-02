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

    def forward(self, task: str, correct_answer: str) -> str:
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        cot_agents = [self.Agent(agent_name=f'Chain-of-Thought Agent {i}', temperature=0.8) for i in range(3)]
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_evaluation_meeting")
        meeting.agents.extend([system] + cot_agents)
        
        # Initial attempt
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please think step by step and then solve the task: {task}"
        ))
        
        # Get initial outputs from all Chain-of-Thought agents
        outputs = []
        for agent in cot_agents:
            output = agent.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            })
            outputs.append(output)
            meeting.chats.append(self.Chat(
                agent=agent,
                content=output["thinking"] + output["answer"]
            ))
        
        # Evaluate outputs and provide feedback
        feedbacks = []
        for i, output in enumerate(outputs):
            feedback = "Your answer is correct." if output["answer"] == correct_answer else "Consider revising your reasoning."
            feedbacks.append(feedback)
            meeting.chats.append(self.Chat(
                agent=cot_agents[i],
                content=feedback
            ))
    
        # Adjust outputs based on feedback
        for i, output in enumerate(outputs):
            adjusted_output = f"{output['thinking']} Based on feedback: {feedbacks[i]}"
            outputs[i]['adjusted_answer'] = adjusted_output
    
        # Final decision based on adjusted outputs
        meeting.chats.append(self.Chat(
            agent=system,
            content="Given all the feedback and adjusted solutions, reason over them carefully and provide a final answer."
        ))
        final_output = final_decision_agent.forward(response_format={
            "thinking": "Your step by step thinking comparing all solutions.",
            "answer": "A single letter, A, B, C or D."
        }, correct_answer=correct_answer)  # Pass the correct answer here
        
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
