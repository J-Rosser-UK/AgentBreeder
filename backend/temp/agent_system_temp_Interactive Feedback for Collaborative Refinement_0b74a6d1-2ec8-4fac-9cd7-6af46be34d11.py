import random
import pandas

from base import Agent, Meeting, Chat

class AgentSystem:
    def forward(self, task: str) -> str:
        # Create agents
        system = Agent(agent_name='system', temperature=0.8)
        principle_agent = Agent(agent_name='Principle Agent', temperature=0.8)
        cot_agent = Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        feedback_agent = Agent(agent_name='Feedback Agent', temperature=0.5)
        
        # Setup meeting
        meeting = Meeting(meeting_name="interactive_feedback_meeting")
        meeting.agents.extend([system, principle_agent, cot_agent, feedback_agent])
        
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
        
        initial_output = cot_agent.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        
        meeting.chats.append(Chat(
            agent=cot_agent,
            content=initial_output["thinking"] + initial_output["answer"]
        ))
        
        # Engage in feedback dialogue
        meeting.chats.append(Chat(
            agent=system,
            content="Evaluate the answer provided by the CoT agent based on the principles and provide specific guidance."
        ))
        
        feedback_output = feedback_agent.forward(response_format={
            "guidance": "Your guidance for refining the CoT agent's output."
        })
        
        meeting.chats.append(Chat(
            agent=feedback_agent,
            content=feedback_output["guidance"]
        ))
        
        # Refine the answer based on feedback
        meeting.chats.append(Chat(
            agent=system,
            content="Given the guidance above, refine your answer: {task}"
        ))
        
        refined_output = cot_agent.forward(response_format={
            "thinking": "Your refined step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        
        return refined_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
