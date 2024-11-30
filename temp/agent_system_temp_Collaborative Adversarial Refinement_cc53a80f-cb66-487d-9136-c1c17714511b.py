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
        # Create multiple reasoning agents
        reasoning_agents = [self.Agent(agent_name=f'Reasoning Agent {i}', temperature=0.8) for i in range(3)]
        # Create adversarial agents
        adversarial_agents = [self.Agent(agent_name=f'Adversarial Agent {i}', temperature=0.8) for i in range(2)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name='collaborative_adversarial_refinement_meeting')
        meeting.agents.extend(reasoning_agents + adversarial_agents)
        
        # Reasoning agents provide initial answers
        meeting.chats.append(self.Chat(
            agent=reasoning_agents[0],
            content=f"Reasoning agents, please analyze and provide your answer for the task: {task}."
        ))
        
        # Collect answers from reasoning agents
        answers = []
        for agent in reasoning_agents:
            output = agent.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C, or D."
            })
            answers.append(output["answer"])
            meeting.chats.append(self.Chat(
                agent=agent,
                content=output["thinking"] + output["answer"]
            ))
        
        # Adversarial agents challenge the reasoning
        for adversary in adversarial_agents:
            for answer in answers:
                meeting.chats.append(self.Chat(
                    agent=adversary,
                    content=f"Critique the answer: {answer}. What are the potential weaknesses?"
                ))
                adversarial_output = adversary.forward(response_format={
                    "critique": "Your critique of the answer and potential weaknesses."
                })
                meeting.chats.append(self.Chat(
                    agent=adversary,
                    content=adversarial_output["critique"]
                ))
                
                # Reasoning agents respond directly to critiques
                for agent in reasoning_agents:
                    meeting.chats.append(self.Chat(
                        agent=agent,
                        content=f"Based on the critique: {adversarial_output['critique']}, please refine your answer."
                    ))
                    refined_output = agent.forward(response_format={
                        "thinking": "Your refined reasoning considering critiques.",
                        "answer": "A single letter, A, B, C, or D."
                    })
                    meeting.chats.append(self.Chat(
                        agent=agent,
                        content=refined_output["thinking"] + refined_output["answer"]
                    ))
        
        # Voting mechanism for final answer
        from collections import Counter
        final_answer = Counter(refined_output["answer"] for agent in reasoning_agents).most_common(1)[0][0]
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
