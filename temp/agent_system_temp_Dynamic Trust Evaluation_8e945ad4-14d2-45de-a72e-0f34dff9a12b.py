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
        # Create a system agent to provide instructions
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create the Chain-of-Thought agent
        cot_agent = self.Agent(
            agent_name='Chain-of-Thought Agent',
            temperature=0.7
        )
        
        # Create a Trust Evaluation agent
        trust_agent = self.Agent(
            agent_name='Trust Evaluation Agent',
            temperature=0.6
        )
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_trust_evaluation")
        meeting.agents.extend([system, cot_agent, trust_agent])
        
        # Add system instruction
        meeting.chats.append(
            self.Chat(
                agent=system, 
                content=f"Please think step by step and then solve the task: {task}"
            )
        )
        
        # Get response from COT agent
        output = cot_agent.forward(
            response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            }
        )
        
        # Record the agent's response in the meeting
        meeting.chats.append(
            self.Chat(
                agent=cot_agent, 
                content=output["thinking"]
            )
        )
        
        # Trust evaluation of the response
        correct_answer = get_correct_answer(task)  # Dynamically get the correct answer based on the task
        trust_output = trust_agent.forward(
            response_format={
                "evaluation": output["answer"] == correct_answer,
                "trust_score": "A score from 0 to 1 indicating reliability."
            }
        )
        
        # Update trust scores based on evaluation
        meeting.chats.append(
            self.Chat(
                agent=trust_agent,
                content=f"Trust score for the response: {trust_output['trust_score']}.",
            )
        )
        
        # Final decision based on trust score
        if trust_output['trust_score'] > 0.7:
            return output["answer"]
        else:
            return "Uncertain response, further review needed."

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
