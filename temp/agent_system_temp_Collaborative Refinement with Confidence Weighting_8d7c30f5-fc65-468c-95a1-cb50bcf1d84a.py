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
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        reasoning_agents = [self.Agent(agent_name=f'Reasoning Agent {i}', temperature=0.8) for i in range(2)]  # Two agents
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_refinement")
        meeting.agents.extend([system] + reasoning_agents)
        
        # Each agent shares their initial thoughts
        outputs = []
        for agent in reasoning_agents:
            meeting.chats.append(self.Chat(
                agent=agent,
                content=f"Please think step by step and provide your thoughts on the task: {task}"
            ))
            output = agent.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C, or D.",
                "confidence": "Your confidence level (0-1)."
            })
            outputs.append(output)
            meeting.chats.append(self.Chat(
                agent=agent,
                content=f"I think the answer is: {output['answer']} with confidence {output['confidence']}"
            ))
        
        # Discussion phase: agents critique each other's reasoning
        meeting.chats.append(self.Chat(
            agent=system,
            content="Agents, please discuss and critique each other's answers."
        ))
        for output in outputs:
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Agent reasoning: {output['thinking']} - Proposed answer: {output['answer']}"
            ))
        
        refined_outputs = []
        for agent in reasoning_agents:
            refined_output = agent.forward(response_format={
                "thinking": "Your refined reasoning after discussion.",
                "answer": "A single letter, A, B, C, or D.",
                "confidence": "Your confidence level (0-1)."
            })
            refined_outputs.append(refined_output)
            meeting.chats.append(self.Chat(
                agent=agent,
                content=f"After discussion, I think the answer is: {refined_output['answer']} with confidence {refined_output['confidence']}"
            ))
        
        # Final consensus decision using confidence weighting
        final_answers = [output['answer'] for output in refined_outputs]
        confidence_scores = [output['confidence'] for output in refined_outputs]  # Use actual confidence scores
        weighted_answers = {answer: 0 for answer in set(final_answers)}
        for answer, score in zip(final_answers, confidence_scores):
            weighted_answers[answer] += score
        final_answer = max(weighted_answers, key=weighted_answers.get)  # Choose the answer with the highest total confidence
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
