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
        # Create system, feedback, and primary agent instances
        system = self.Agent(agent_name='system', temperature=0.8)
        feedback_agent = self.Agent(agent_name='Feedback Agent', temperature=0.7)
        population_size = 5  # Number of primary agents
        agents = [self.Agent(agent_name=f'Agent {i}', temperature=0.7) for i in range(population_size)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_dialogue")
        meeting.agents.extend([system, feedback_agent] + agents)
        
        # Step 1: Individual reasoning
        outputs = []
        for agent in agents:
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Agent {agent.agent_name}, please solve the task: {task}"
                )
            )
            output = agent.forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C, or D."
                }
            )
            outputs.append(output)
            meeting.chats.append(
                self.Chat(
                    agent=agent, 
                    content=output["thinking"]
                )
            )
    
        # Step 2: Sharing and discussing outputs
        for i, agent in enumerate(agents):
            for j, other_agent in enumerate(agents):
                if i != j:
                    meeting.chats.append(
                        self.Chat(
                            agent=feedback_agent,
                            content=f"Agent {other_agent.agent_name}, please critique the reasoning of Agent {agent.agent_name}."
                        )
                    )
                    meeting.chats.append(
                        self.Chat(
                            agent=other_agent,
                            content=f"I think your reasoning is {outputs[i][\"thinking\"]}. What do you think about it?"
                        )
                    )
                    feedback = feedback_agent.forward(
                        response_format={
                            "feedback": "Your evaluation of the discussion."
                        }
                    )
                    meeting.chats.append(
                        self.Chat(
                            agent=system,
                            content=feedback["feedback"]
                        )
                    )
    
        # Step 3: Final consensus
        final_answers = [output["answer"] for output in outputs]
        # Use majority voting for final answer without manual aggregation
        return max(set(final_answers), key=final_answers.count) if final_answers else "No answer"

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
