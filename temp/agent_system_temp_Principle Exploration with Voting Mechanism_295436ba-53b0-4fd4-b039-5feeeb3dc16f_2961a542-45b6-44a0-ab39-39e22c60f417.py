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
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        principle_agents = [self.Agent(agent_name=f'Principle Agent {i+1}', temperature=0.8) for i in range(3)]
        cot_agents = [self.Agent(agent_name=f'Chain-of-Thought Agent {i+1}', temperature=0.8) for i in range(2)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="principle_exploration_meeting")
        meeting.agents.extend([system] + principle_agents + cot_agents)
        
        # First get the principles involved
        meeting.chats.append(self.Chat(
            agent=system,
            content="What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
        ))
        
        for agent in principle_agents:
            principle_output = agent.forward(response_format={
                "thinking": "Your step by step thinking about the principles.",
                "principles": "List and explanation of the principles involved."
            })
            meeting.chats.append(self.Chat(
                agent=agent,
                content=principle_output["thinking"] + principle_output["principles"]
            ))
        
        # Now solve using the principles
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the question and the involved principles above, think step by step and then solve the task: {task}"
        ))
        
        # Aggregate final outputs using majority voting directly
        answer_counts = {}
        for agent in cot_agents:
            final_output = agent.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            })
            meeting.chats.append(self.Chat(
                agent=agent,
                content=final_output["thinking"]
            ))
            answer = final_output["answer"]
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # Determine the final answer based on counts
        final_answer = max(answer_counts, key=answer_counts.get)
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
