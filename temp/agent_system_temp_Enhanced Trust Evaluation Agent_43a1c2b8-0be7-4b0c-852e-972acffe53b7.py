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
        trust_evaluator = self.Agent(agent_name='Trust Evaluation Agent', temperature=0.6)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="trust_evaluation_meeting")
        meeting.agents.extend([system] + cot_agents + [trust_evaluator])
        
        # Add system instruction
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please think step by step and then solve the task: {task}"
        ))
        
        # Collect answers from all COT agents
        answers = []
        for agent in cot_agents:
            output = agent.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            })
            answers.append((output["answer"], output["thinking"]))
            meeting.chats.append(self.Chat(agent=agent, content=output["thinking"]))
    
        # Trust evaluation
        trust_scores = []
        for answer, thinking in answers:
            trust_output = trust_evaluator.forward(response_format={
                "feedback": f"Rate the following thinking: {thinking} and answer: {answer} on a scale of 1-5 for clarity, relevance, and correctness."
            })
            trust_scores.append((answer, trust_output["feedback"].strip()))  # Store score
            meeting.chats.append(self.Chat(agent=trust_evaluator, content=trust_output["feedback"]))
    
        # Determine the most trusted answer based on scores
        best_answer = max(trust_scores, key=lambda x: int(x[1]))[0]  # Select the answer with the highest trust score
        return best_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
