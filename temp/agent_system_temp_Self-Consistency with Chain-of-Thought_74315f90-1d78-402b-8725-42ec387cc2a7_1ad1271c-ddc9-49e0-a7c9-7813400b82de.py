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
        # Create a system agent to provide instructions
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create multiple CoT agents with higher temperature for varied reasoning
        N = 3  # Number of CoT agents
        cot_agents = [
            self.Agent(
                agent_name=f'Chain-of-Thought Agent {i}',
                temperature=0.8
            ) for i in range(N)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="self-consistency")
        meeting.agents.extend([system] + cot_agents)
        
        # Collect answers from all agents
        possible_answers = []
        for i in range(N):
            # Add system instruction
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Please think step by step and then solve the task: {task}"
                )
            )
            
            # Get response from current COT agent
            output = cot_agents[i].forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            
            # Record the agent's response
            meeting.chats.append(
                self.Chat(
                    agent=cot_agents[i], 
                    content=output["thinking"]
                )
            )
            
            possible_answers.append(output["answer"])
        
        # Select the most common answer through majority voting
        from collections import Counter
        
        final_answer = Counter(possible_answers).most_common(1)[0][0]
        return final_answer
    

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
