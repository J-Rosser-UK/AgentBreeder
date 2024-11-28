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
        # Create a system agent to oversee coalition formation
        system = self.Agent(agent_name='system', temperature=0.8)
        
        # Create agents with different expertise
        physics_agent = self.Agent(agent_name='Physics Expert', temperature=0.8)
        chemistry_agent = self.Agent(agent_name='Chemistry Expert', temperature=0.8)
        biology_agent = self.Agent(agent_name='Biology Expert', temperature=0.8)
        
        # Setup initial meeting for coalition formation
        meeting = self.Meeting(meeting_name="coalition_formation")
        meeting.agents.extend([system, physics_agent, chemistry_agent, biology_agent])
        
        # System instructs agents to propose coalitions and negotiate
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the task: {task}, please propose coalitions based on your strengths and negotiate with others."
        ))
        
        # Each agent proposes a coalition and negotiates
        coalition_proposals = []
        for agent in meeting.agents[1:]:  # Skip system agent
            proposal_output = agent.forward(response_format={
                "proposal": "Your coalition proposal (e.g., 'I will work with Chemistry Expert')"
            })
            coalition_proposals.append(proposal_output["proposal"])
            meeting.chats.append(self.Chat(
                agent=agent,
                content=proposal_output["proposal"]
            ))
        
        # Form coalitions based on negotiations
        coalitions = []
        if "Physics Expert" in coalition_proposals and "Chemistry Expert" in coalition_proposals:
            coalitions.append([physics_agent, chemistry_agent])
        if "Biology Expert" in coalition_proposals:
            coalitions.append([biology_agent])  # Biology agent works alone for now
        
        # Each coalition works on the task
        coalition_outputs = []
        for coalition in coalitions:
            coalition_meeting = self.Meeting(meeting_name="coalition_work")
            coalition_meeting.agents.extend(coalition)
            coalition_meeting.chats.append(self.Chat(
                agent=system,
                content=f"Coalition working on task: {task}. Please collaborate and solve it step by step."
            ))
            
            # Get coalition answers
            coalition_output = [agent.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C, or D."
            }) for agent in coalition if agent is not None]
            
            # Ensure coalition_output is not empty before appending
            if coalition_output:
                coalition_outputs.append(coalition_output)
                
                # Log coalition outputs
                for output in coalition_output:
                    coalition_meeting.chats.append(self.Chat(
                        agent=coalition[0],  # Assume first agent logs the response
                        content=output["thinking"] + output["answer"]
                    ))
            else:
                meeting.chats.append(self.Chat(
                    agent=system,
                    content="No valid outputs from this coalition."
                ))
        
        # Evaluate responses and select the most common answer from coalitions
        from collections import Counter
        all_answers = [output[0]["answer"] for output in coalition_outputs if output]
        if all_answers:
            final_answer = Counter(all_answers).most_common(1)[0][0]
            return final_answer
        else:
            return "No valid answer generated."

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
