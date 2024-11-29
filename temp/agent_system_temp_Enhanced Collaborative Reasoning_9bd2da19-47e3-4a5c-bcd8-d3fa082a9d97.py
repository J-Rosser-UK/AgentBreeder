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

    import numpy as np
    
    class EnhancedCollaborativeReasoning:
        def forward(self, task: str) -> str:
            # Create agents
            system = Agent(agent_name='system', temperature=0.8)
            principle_agent = Agent(agent_name='Principle Agent', temperature=0.8)
            reasoning_agents = [Agent(agent_name=f'Reasoning Agent {i+1}', temperature=0.8) for i in range(2)]
            
            # Setup meeting
            meeting = Meeting(meeting_name="enhanced_collaborative_reasoning_meeting")
            meeting.agents.extend([system, principle_agent] + reasoning_agents)
            
            # Get the principles involved
            meeting.chats.append(Chat(
                agent=system,
                content="What principles are involved in solving this task? Please think step by step and explain your reasoning."
            ))
            
            principle_output = principle_agent.forward(response_format={
                "thinking": "Your step by step thinking about the principles.",
                "principles": "List and explanation of the principles involved."
            })
            
            meeting.chats.append(Chat(
                agent=principle_agent,
                content=principle_output["thinking"] + principle_output["principles"]
            ))
            
            # Reasoning agents share their thoughts
            answers = []
            for reasoning_agent in reasoning_agents:
                meeting.chats.append(Chat(
                    agent=reasoning_agent,
                    content=f"Based on the principles discussed, please share your reasoning for the task: {task}"
                ))
                reasoning_output = reasoning_agent.forward(response_format={
                    "thinking": "Your step by step reasoning.",
                    "answer": "A single letter, A, B, C or D."
                })
                meeting.chats.append(Chat(
                    agent=reasoning_agent,
                    content=reasoning_output["thinking"] + reasoning_output["answer"]
                ))
                
                # Introduce structured noise to simulate uncertainty
                noisy_answer = self.introduce_noise(reasoning_output["answer"])
                answers.append(noisy_answer)
                meeting.chats.append(Chat(
                    agent=system,
                    content=f"Agent {reasoning_agent.agent_name} proposed: {noisy_answer}"
                ))
            
            # Final decision based on all inputs using majority voting
            final_answer = self.determine_final_answer(answers)
            return final_answer
        
        def introduce_noise(self, answer: str) -> str:
            # Introduce noise with a certain probability
            return answer if np.random.rand() > 0.2 else np.random.choice(['A', 'B', 'C', 'D'])
        
        def determine_final_answer(self, answers: list) -> str:
            from collections import Counter
            return Counter(answers).most_common(1)[0][0]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
