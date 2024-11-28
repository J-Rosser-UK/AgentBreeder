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
        reasoning_agents = [self.Agent(agent_name=f'Reasoning Agent {i}', temperature=0.7) for i in range(3)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_reasoning")
        meeting.agents.extend([system] + reasoning_agents)
        
        # Instruction for reasoning agents
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Agents, please think step by step about the task: {task} and share your reasoning with a confidence score from 0 to 1."
        ))
        
        # Collect responses from reasoning agents with confidence score
        responses = []
        for agent in reasoning_agents:
            output = agent.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C, or D.",
                "confidence": "A score between 0 and 1 indicating your confidence."
            })
            
            # Introduce controlled noise based on confidence
            noise_adjustment = np.random.normal(0, 0.1) * output["confidence"]
            noisy_answer = output["answer"] if np.random.rand() > noise_adjustment else np.random.choice(['A', 'B', 'C', 'D'])
            responses.append((noisy_answer, output["confidence"]))
            meeting.chats.append(self.Chat(
                agent=agent,
                content=output["thinking"] + f" (Adjusted Output: {noisy_answer}, Confidence: {output['confidence']})"
            ))
        
        # Aggregate responses to determine the final answer based on confidence
        weighted_responses = Counter()
        for answer, confidence in responses:
            weighted_responses[answer] += confidence
        final_answer = weighted_responses.most_common(1)[0][0]
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
