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
            self.Agent(agent_name=f'Chain-of-Thought Agent {i+1}', temperature=0.8) for i in range(3)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="enhanced_debate_chain_of_thought_meeting")
        meeting.agents.extend([system] + cot_agents)
        
        # Get principles involved from the system agent
        meeting.chats.append(self.Chat(
            agent=system,
            content="What are the relevant principles and concepts involved in solving this task? Please list and explain them."
        ))
        
        principle_agent = self.Agent(agent_name='Principle Agent', temperature=0.8)
        principle_output = principle_agent.forward(response_format={
            "thinking": "Your step by step thinking about the principles.",
            "principles": "List and explanation of the principles involved."
        })
        meeting.chats.append(self.Chat(
            agent=principle_agent,
            content=principle_output["thinking"] + principle_output["principles"]
        ))
        
        # Reasoning phase with debate
        final_outputs = []
        for agent in cot_agents:
            meeting.chats.append(self.Chat(
                agent=agent,
                content="Given the principles above, please discuss your reasoning for solving the task. You have a limit of 100 tokens."
            ))
            output = agent.forward(response_format={
                "thinking": "Your step by step thinking within the token limit.",
                "answer": "A single letter, A, B, C, or D."
            })
            final_outputs.append(output["answer"])
            meeting.chats.append(self.Chat(
                agent=agent,
                content=output["thinking"] + f' Answer: {output["answer"]}'
            ))
        
        # Debate phase: critique each other's answers
        for i, agent in enumerate(cot_agents):
            for j, critic in enumerate(cot_agents):
                if i != j:
                    meeting.chats.append(self.Chat(
                        agent=agent,
                        content=f"Critique the answer from {critic.agent_name}: {final_outputs[j]}"
                    ))
                    critique_output = critic.forward(response_format={
                        "critique": "Your critique of the answer and reasoning."
                    })
                    meeting.chats.append(self.Chat(
                        agent=critic,
                        content=critique_output["critique"]
                    ))
        
        # Aggregate answers from all cot agents by majority voting
        final_answer = max(set(final_outputs), key=final_outputs.count)
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
