import random
import pandas

from base import Agent, Meeting, Chat

from sqlalchemy import Session

class AgentSystem:
    def __init__(self, session: Session):
        self.Agent = Wrapper(Agent, session)
        self.Meeting = Wrapper(Meeting, session)
        self.Chat = Wrapper(Chat, session)
        self.session = session

    def forward(self, task: str) -> str:
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        physics_expert = self.Agent(agent_name='Physics Expert', temperature=0.8)
        chemistry_expert = self.Agent(agent_name='Chemistry Expert', temperature=0.8)
        biology_expert = self.Agent(agent_name='Biology Expert', temperature=0.8)
    
        # Setup meeting for coalition
        meeting = self.Meeting(meeting_name="negotiable_coalition_meeting")
        meeting.agents.extend([system, physics_expert, chemistry_expert, biology_expert])
    
        # Coalition formation step
        meeting.chats.append(self.Chat(
            agent=system,
            content="Given the task, let's discuss how we can best collaborate to solve it. What roles should we take?"
        ))
    
        # Propose roles and strategies
        role_outputs = []
        for agent in [physics_expert, chemistry_expert, biology_expert]:
            role_output = agent.forward(response_format={
                "role_proposal": "Propose your role in the coalition.",
                "strategy": "Your strategy for solving the task."
            })
            role_outputs.append(role_output)
            meeting.chats.append(self.Chat(
                agent=agent,
                content=role_output["role_proposal"] + " - Strategy: " + role_output["strategy"]
            ))
    
        # Negotiation phase: agents discuss and refine their roles
        meeting.chats.append(self.Chat(
            agent=system,
            content="Let’s discuss and refine our roles based on the proposals."
        ))
        for output in role_outputs:
            meeting.chats.append(self.Chat(
                agent=system,
                content="Agent's proposal: " + output["role_proposal"] + ". What do others think?"
            ))
            for agent in [physics_expert, chemistry_expert, biology_expert]:
                feedback = agent.forward(response_format={
                    "feedback": "Provide your feedback on the proposal."
                })
                meeting.chats.append(self.Chat(
                    agent=agent,
                    content="Feedback on proposal: " + feedback["feedback"]
                ))
    
        # Each agent will contribute based on their agreed role
        task_outputs = []
        for agent in [physics_expert, chemistry_expert, biology_expert]:
            task_output = agent.forward(response_format={
                "thinking": "Your step by step reasoning based on your role.",
                "answer": "A single letter, A, B, C, or D."
            })
            task_outputs.append(task_output)
            meeting.chats.append(self.Chat(
                agent=agent,
                content=task_output["thinking"] + " - Answer: " + task_output["answer"]
            ))
    
        # Final decision based on coalition's reasoning
        meeting.chats.append(self.Chat(
            agent=system,
            content="Given all the contributions, let’s decide on the final answer."
        ))
        final_output = system.forward(response_format={
            "final_thinking": "Your reasoning for the final decision.",
            "final_answer": "A single letter, A, B, C, or D."
        })
    
        return final_output["final_answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
