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
        principle_agent = self.Agent(agent_name='Principle Agent', temperature=0.8)
        cot_agent_1 = self.Agent(agent_name='Chain-of-Thought Agent 1', temperature=0.8)
        cot_agent_2 = self.Agent(agent_name='Chain-of-Thought Agent 2', temperature=0.8)
        trust_agent = self.Agent(agent_name='Trust Evaluation Agent', temperature=0.7)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_chain_of_thought_with_adaptive_trust")
        meeting.agents.extend([system, principle_agent, cot_agent_1, cot_agent_2, trust_agent])
        
        # Get principles involved from the principle agent
        meeting.chats.append(self.Chat(
            agent=system,
            content="What are the relevant principles and concepts involved in solving this task? Please list and explain them."
        ))
        
        principle_output = principle_agent.forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        meeting.chats.append(self.Chat(
            agent=principle_agent,
            content=principle_output["thinking"] + principle_output["principles"]
        ))
        
        # Each CoT agent discusses their reasoning based on the principles
        meeting.chats.append(self.Chat(
            agent=cot_agent_1,
            content="Given the principles above, discuss your reasoning for solving the task."
        ))
        meeting.chats.append(self.Chat(
            agent=cot_agent_2,
            content="Given the principles above, discuss your reasoning for solving the task."
        ))
        
        # Now solve using the principles with both CoT agents
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the question and the involved principles above, think step by step and then solve the task: {task}"
        ))
        
        final_output_1 = cot_agent_1.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        final_output_2 = cot_agent_2.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        
        # Trust evaluation
        meeting.chats.append(self.Chat(
            agent=trust_agent,
            content="Evaluate the reliability of the following responses based on logical consistency and relevance: \n1. Response from Agent 1: {final_output_1['answer']} \n2. Response from Agent 2: {final_output_2['answer']}"
        ))
        trust_output = trust_agent.forward(response_format={
            "trust_scores": "Trust scores for each response, format: {response_1: score, response_2: score}"
        })
        
        # Ensure trust_output is correctly structured
        if isinstance(trust_output, dict) and "trust_scores" in trust_output:
            trust_scores = trust_output["trust_scores"]
        else:
            raise ValueError("Trust output is not in the expected format.")
        
        # Aggregate answers based on trust scores using weighted voting
        responses = [final_output_1["answer"], final_output_2["answer"]]
        trusted_answer = responses[0] if trust_scores.get(responses[0], 0) >= trust_scores.get(responses[1], 0) else responses[1]
        
        return trusted_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
