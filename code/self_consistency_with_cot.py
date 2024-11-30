import random
import numpy as np
import pandas

from base import Agent, Meeting, Chat, Wrapper

from sqlalchemy.orm import Session


class SelfConsistencyCoT:
    
    def __init__(self, session: Session):
        self.Agent = Wrapper(Agent, session)
        self.Meeting = Wrapper(Meeting, session)
        self.Chat = Wrapper(Chat, session)
        self.session = session

    def forward(self, task: str = "What should I have for dinner? A: soup, B: pasta, C: burger") -> str:
    
        system = self.Agent(agent_name='system')
        cot_agents = [self.Agent(agent_name=f'Chain-of-Thought Agent {i}') for i in range(5)]
        
        meeting = self.Meeting(meeting_name="self-consistency")
        meeting.agents.extend([system] + cot_agents)
        
        possible_answers = []
        for agent in cot_agents:
            meeting.chats.append(self.Chat(agent=system, content=f"Think and then solve the task: {task}"))
            output = agent.forward(response_format={"thinking": "Your thinking.", "answer": "A, B or C"})
            meeting.chats.append(self.Chat(agent=agent, content=output["thinking"]))
            possible_answers.append(output["answer"])

        from collections import Counter
        final_answer = Counter(possible_answers).most_common(1)[0][0]
        return final_answer # A


if __name__ == '__main__':
    sc_system = SelfConsistencyCoT()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = sc_system.forward(task)
    print(output)