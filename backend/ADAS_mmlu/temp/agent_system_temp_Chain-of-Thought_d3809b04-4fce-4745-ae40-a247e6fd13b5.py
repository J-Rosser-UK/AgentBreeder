import random
import pandas

from agent import Agent, Meeting, Chat

class AgentSystem:
    def forward(self, task: str) -> str:
        # Create a system agent to provide instructions
        system = Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create the Chain-of-Thought agent
        cot_agent = Agent(
            agent_name='Chain-of-Thought Agent',
            temperature=0.7
        )
        
        # Setup meeting
        meeting = Meeting(meeting_name="chain-of-thought")
        meeting.agents.extend([system, cot_agent])
        
        # Add system instruction
        meeting.chats.append(
            Chat(
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
            Chat(
                agent=cot_agent, 
                content=output["thinking"]
            )
        )
        
        return output["answer"]
    