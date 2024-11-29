import random
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
        
        # Create a Bias Detection Agent
        bias_agent = self.Agent(
            agent_name='Bias Detection Agent',
            temperature=0.6
        )
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="bias_aware_chain_of_thought")
        meeting.agents.extend([system, cot_agent, bias_agent])
        
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
        
        # Bias assessment
        meeting.chats.append(
            self.Chat(
                agent=system,
                content="Please analyze the answer above for potential biases and suggest adjustments if needed."
            )
        )
        
        bias_output = bias_agent.forward(
            response_format={
                "feedback": "Your detailed feedback on biases.",
                "adjusted_answer": "A single letter, A, B, C or D."
            }
        )
        
        # Record bias agent's feedback
        meeting.chats.append(
            self.Chat(
                agent=bias_agent,
                content=bias_output["feedback"]
            )
        )
        
        # Return the adjusted answer based on the bias agent's evaluation
        return bias_output["adjusted_answer"]
    

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
