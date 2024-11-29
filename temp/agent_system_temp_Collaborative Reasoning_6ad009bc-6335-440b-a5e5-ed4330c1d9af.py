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
        agents = [
            self.Agent(agent_name=f'Reasoning Agent {i}', temperature=0.7)
            for i in range(3)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_reasoning")
        meeting.agents.extend(agents)
        
        # Initial reasoning phase
        for agent in agents:
            meeting.chats.append(
                self.Chat(
                    agent=agent,
                    content=f"Please think step by step and provide your reasoning for the task: {task}"
                )
            )
            
            # Get response from each agent
            output = agent.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C, or D."
            })
            
            # Record the agent's initial reasoning in the meeting
            meeting.chats.append(
                self.Chat(
                    agent=agent,
                    content=output["thinking"] + f' Answer: {output["answer"]}'
                )
            )
        
        # Discussion phase: agents can challenge each other's reasoning
        for round in range(3):  # Allow for multiple rounds of discussion
            for i, agent in enumerate(agents):
                # Allow each agent to ask questions to another agent
                question_agent = agents[(i + 1) % len(agents)]
                meeting.chats.append(
                    self.Chat(
                        agent=agent,
                        content=f"{question_agent.agent_name}, can you clarify your reasoning?"
                    )
                )
                question_output = question_agent.forward(response_format={
                    "clarification": "Your clarification on the reasoning."
                })
                meeting.chats.append(
                    self.Chat(
                        agent=question_agent,
                        content=question_output["clarification"]
                    )
                )
        
        # Final consensus phase
        final_answers = []
        for agent in agents:
            final_output = agent.forward(response_format={
                "final_thinking": "Your final reasoning after discussion.",
                "final_answer": "A single letter, A, B, C, or D."
            })
            final_answers.append(final_output["final_answer"])
        
        # Return the final answer (could use majority vote or consensus)
        from collections import Counter
        final_answer = Counter(final_answers).most_common(1)[0][0]
        return final_answer
    

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
