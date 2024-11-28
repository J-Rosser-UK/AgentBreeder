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
        expert_agents = [
            self.Agent(agent_name='Physics Expert', temperature=0.8),
            self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            self.Agent(agent_name='Biology Expert', temperature=0.8)
        ]
        
        # Setup meeting for coalition formation
        meeting = self.Meeting(meeting_name="negotiated_coalition")
        meeting.agents.extend([system] + expert_agents)
        
        # Each agent evaluates potential allies and proposes coalition structures
        coalition_proposals = []
        for agent in expert_agents:
            meeting.chats.append(self.Chat(
                agent=agent,
                content="Given the task, evaluate your strengths and weaknesses and suggest potential coalition members."
            ))
            proposal_output = agent.forward(response_format={
                "proposal": "Your proposed coalition members."
            })
            coalition_proposals.append(proposal_output["proposal"])
        
        # Form coalitions based on proposals
        coalition_members = []
        for proposal in coalition_proposals:
            for member_name in proposal:
                # Find the actual agent instance from the expert_agents list
                member_agent = next((a for a in expert_agents if a.agent_name == member_name), None)
                if member_agent:
                    coalition_members.append(member_agent)  # Collect actual agent instances
        coalition_members = list(set(coalition_members))  # Remove duplicates
        
        # Coalition negotiation phase
        for member in coalition_members:
            meeting.chats.append(self.Chat(
                agent=member,
                content="We will collaborate to solve the task. Let's discuss our approach."
            ))
        
        # Collaborative problem solving
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the task: {task}, please collaborate and provide a solution."
        ))
        
        # Each coalition member provides their reasoning
        coalition_outputs = []
        for member in coalition_members:
            output = member.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            })
            coalition_outputs.append(output)
            meeting.chats.append(self.Chat(
                agent=member,
                content=output["thinking"]
            ))
        
        # Final decision based on consensus of coalition outputs
        answers = [output["answer"] for output in coalition_outputs]
        final_answer = max(set(answers), key=answers.count)  # Majority voting
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
