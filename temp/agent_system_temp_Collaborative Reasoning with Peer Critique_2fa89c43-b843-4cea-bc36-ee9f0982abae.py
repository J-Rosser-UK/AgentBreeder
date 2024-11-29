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
        cot_agents = [
            self.Agent(agent_name=f'Chain-of-Thought Agent {i}', temperature=0.7)
            for i in range(3)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_reasoning")
        meeting.agents.extend([system] + cot_agents)
        
        # Initial instruction
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please discuss the task and your reasoning process step by step. Task: {task}"
        ))
        
        # Collect reasoning and confidence from each agent
        outputs = []
        for agent in cot_agents:
            output = agent.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C, or D.",
                "confidence": "Your confidence score (0-1)"
            })
            
            # Append structured output
            outputs.append(output)
            meeting.chats.append(self.Chat(
                agent=agent,
                content=output["thinking"] + f" (Answer: {output["answer"]}, Confidence: {output["confidence"]})"
            ))
        
        # Determine the final answer based on weighted voting
        from collections import Counter
        total_confidence = sum(float(output["confidence"]) for output in outputs)
        answer_scores = Counter()
        for output in outputs:
            answer = output["answer"]
            confidence = float(output["confidence"]) / total_confidence
            answer_scores[answer] += confidence
        final_answer = answer_scores.most_common(1)[0][0]
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
