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
        principle_agent_1 = self.Agent(agent_name='Principle Agent 1', temperature=0.8)
        principle_agent_2 = self.Agent(agent_name='Principle Agent 2', temperature=0.8)
        cot_agent_1 = self.Agent(agent_name='Chain-of-Thought Agent 1', temperature=0.7)
        cot_agent_2 = self.Agent(agent_name='Chain-of-Thought Agent 2', temperature=0.9)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_multi_agent_meeting")
        meeting.agents.extend([system, principle_agent_1, principle_agent_2, cot_agent_1, cot_agent_2])
        
        # First get the principles involved from both agents
        meeting.chats.append(self.Chat(
            agent=system,
            content="What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
        ))
        
        principle_output_1 = principle_agent_1.forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        
        principle_output_2 = principle_agent_2.forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        
        # Combine outputs from both principle agents
        combined_principles = principle_output_1["thinking"] + principle_output_1["principles"] + 
                                principle_output_2["thinking"] + principle_output_2["principles"]
        
        meeting.chats.append(self.Chat(
            agent=system,
            content=combined_principles
        ))
        
        # Now solve using the principles with both COT agents
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the question and the involved principles above, think step by step and then solve the task: {task}"
        ))
        
        final_output_1 = cot_agent_1.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        
        final_output_2 = cot_agent_2.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        
        # Implement a consensus mechanism to determine the final answer
        if final_output_1["answer"] == final_output_2["answer"]:
            return final_output_1["answer"]
        else:
            # Ask for clarification if answers differ
            meeting.chats.append(self.Chat(
                agent=system,
                content="The answers from both agents differ. Please clarify your reasoning and provide a consensus answer."
            ))
            clarification_output_1 = cot_agent_1.forward(response_format={
                "clarification": "Please clarify your reasoning."
            })
            clarification_output_2 = cot_agent_2.forward(response_format={
                "clarification": "Please clarify your reasoning."
            })
            
            # Return a structured response indicating the need for clarification
            return json.dumps({"clarification_needed": True, "clarifications": [clarification_output_1["clarification"], clarification_output_2["clarification"]]})

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
