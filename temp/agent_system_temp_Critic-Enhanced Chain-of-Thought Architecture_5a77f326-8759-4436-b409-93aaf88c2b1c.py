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
        cot_agent1 = self.Agent(agent_name='Chain-of-Thought Agent 1', temperature=0.8)
        cot_agent2 = self.Agent(agent_name='Chain-of-Thought Agent 2', temperature=0.8)
        critic_agent = self.Agent(agent_name='Critic Agent', temperature=0.6)
        final_decision_agent1 = self.Agent(agent_name='Final Decision Agent 1', temperature=0.1)
        final_decision_agent2 = self.Agent(agent_name='Final Decision Agent 2', temperature=0.1)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="critic_enhanced_chain_of_thought_meeting")
        meeting.agents.extend([system, cot_agent1, cot_agent2, critic_agent, final_decision_agent1, final_decision_agent2])
        
        N_max = 3  # Maximum number of attempts
        
        # Initial attempt
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Solve the task: {task} step by step."
        ))
        
        # Get outputs from Chain-of-Thought agents
        output1 = cot_agent1.forward(response_format={
            "thinking": "Your step by step reasoning.",
            "answer": "A, B, C, or D."
        })
        output2 = cot_agent2.forward(response_format={
            "thinking": "Your step by step reasoning.",
            "answer": "A, B, C, or D."
        })
        
        # Critique the answers
        meeting.chats.append(self.Chat(
            agent=critic_agent,
            content=f"Critique the answers: {output1['answer']} and {output2['answer']} with reasoning: {output1['thinking']} and {output2['thinking']}"
        ))
        critique_output = critic_agent.forward(response_format={
            "critique": "Your critique of the previous answers."
        })
        meeting.chats.append(self.Chat(
            agent=critic_agent,
            content=critique_output["critique"]
        ))
        
        # Generate diverse solutions
        for i in range(N_max):
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Suggest a new approach for: {task}."
            ))
            
            output1 = cot_agent1.forward(response_format={
                "thinking": "Your new reasoning.",
                "answer": "A, B, C, or D."
            })
            meeting.chats.append(self.Chat(
                agent=cot_agent1,
                content=output1["thinking"] + output1["answer"]
            ))
            
            output2 = cot_agent2.forward(response_format={
                "thinking": "Your new reasoning.",
                "answer": "A, B, C, or D."
            })
            meeting.chats.append(self.Chat(
                agent=cot_agent2,
                content=output2["thinking"] + output2["answer"]
            ))
        
        # Make final decision
        meeting.chats.append(self.Chat(
            agent=system,
            content="Review all solutions and critiques, then provide a final answer."
        ))
        final_output1 = final_decision_agent1.forward(response_format={
            "thinking": "Your reasoning based on all critiques and solutions.",
            "answer": "A, B, C, or D."
        })
        final_output2 = final_decision_agent2.forward(response_format={
            "thinking": "Your reasoning based on all critiques and solutions.",
            "answer": "A, B, C, or D."
        })
        
        return final_output1["answer"]  # or final_output2["answer"] for redundancy.

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
