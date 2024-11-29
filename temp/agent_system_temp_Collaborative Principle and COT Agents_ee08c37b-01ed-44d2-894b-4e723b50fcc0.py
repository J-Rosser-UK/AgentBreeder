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
        principle_agents = [self.Agent(agent_name='Principle Agent', temperature=0.8) for _ in range(2)]
        cot_agents = [self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.8) for _ in range(2)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_step_back_meeting")
        meeting.agents.extend([system] + principle_agents + cot_agents)
        
        # First get the principles involved from all principle agents
        principles_summary = []
        for principle_agent in principle_agents:
            meeting.chats.append(self.Chat(
                agent=system,
                content="What are the physics, chemistry or biology principles and concepts involved in solving this task? First think step by step. Then list all involved principles and explain them."
            ))
            principle_output = principle_agent.forward(response_format={
                "thinking": "Your step by step thinking about the principles.",
                "principles": "List and explanation of the principles involved."
            })
            principles_summary.append(principle_output["principles"])
            meeting.chats.append(self.Chat(
                agent=principle_agent,
                content=principle_output["thinking"] + principle_output["principles"]
            ))
        
        # Now solve using the principles with all cot agents
        answers = []
        for cot_agent in cot_agents:
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Given the question and the involved principles above: {', '.join(principles_summary)}, think step by step and then solve the task: {task}"
            ))
            final_output = cot_agent.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            })
            meeting.chats.append(self.Chat(
                agent=cot_agent,
                content=final_output["thinking"] + final_output["answer"]
            ))
            answers.append(final_output["answer"])
        
        # Implement voting mechanism for answers
        from collections import Counter
        final_answer = Counter(answers).most_common(1)[0][0]
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
