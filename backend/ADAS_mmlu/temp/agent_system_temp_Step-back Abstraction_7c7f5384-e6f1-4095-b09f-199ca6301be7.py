import random
import pandas

from agent import Agent, Meeting, Chat

class AgentSystem:
    def forward(self, task: str) -> str:
        # Create agents
        system = Agent(agent_name='system', temperature=0.8)
        principle_agent = Agent(agent_name='Principle Agent', temperature=0.8)
        cot_agent = Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        
        # Setup meeting
        meeting = Meeting(meeting_name="step_back_meeting")
        meeting.agents.extend([system, principle_agent, cot_agent])
        
        # First get the principles involved
        meeting.chats.append(Chat(
            agent=system,
            content="What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
        ))
        
        principle_output = principle_agent.forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        
        meeting.chats.append(Chat(
            agent=principle_agent,
            content=principle_output["thinking"] + principle_output["principles"]
        ))
        
        # Now solve using the principles
        meeting.chats.append(Chat(
            agent=system,
            content=f"Given the question and the involved principles above, think step by step and then solve the task: {task}"
        ))
        
        final_output = cot_agent.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        
        return final_output["answer"]
    