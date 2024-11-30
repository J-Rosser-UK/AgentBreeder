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
        cot_agents = [self.Agent(agent_name=f'Chain-of-Thought Agent {i}', temperature=0.8) for i in range(3)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_role_allocation_with_peer_review")
        meeting.agents.extend([system] + cot_agents)
        
        # Initial attempts to solve the task
        outputs = []
        for agent in cot_agents:
            meeting.chats.append(self.Chat(
                agent=agent,
                content=f"Please think step by step and then solve the task: {task}"
            ))
            output = agent.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C, or D."
            })
            outputs.append((agent.agent_name, output))
            meeting.chats.append(self.Chat(
                agent=agent,
                content=output["thinking"] + output["answer"]
            ))
        
        # Evaluate agent performance
        performance_scores = {agent.agent_name: {\'correct\': 0, \'clarity\': 0, \'depth\': 0} for agent in cot_agents}
        for agent_name, output in outputs:
            if output["answer"] in ["A", "B", "C", "D"]:
                performance_scores[agent_name][\'correct\'] += 1  # Score for valid answer
            performance_scores[agent_name][\'clarity\'] += evaluate_clarity(output["thinking"])
            performance_scores[agent_name][\'depth\'] += evaluate_depth(output["thinking"])
    
        # Assign roles based on performance
        sorted_agents = sorted(performance_scores.items(), key=lambda x: (x[1][\'correct\'], x[1][\'clarity\'], x[1][\'depth\']), reverse=True)
        lead_agent_name = sorted_agents[0][0]  # Top performer
        support_agents = [name for name, _ in sorted_agents[1:]]  # Others as support
    
        # Peer review phase
        for agent in cot_agents:
            meeting.chats.append(self.Chat(
                agent=agent,
                content="Please review the reasoning of the Lead Reasoner and provide feedback."
            ))
            peer_feedback = agent.forward(response_format={
                "feedback": "Your feedback on the reasoning."
            })
            meeting.chats.append(self.Chat(
                agent=agent,
                content=peer_feedback["feedback"]
            ))
    
        # Perform a second round of reasoning with dynamic roles
        meeting.chats.append(self.Chat(
            agent=system,
            content="Now, based on the performance and feedback, the top agent will lead the reasoning and others will support."
        ))
        
        meeting.chats.append(self.Chat(
            agent=cot_agents[0],
            content=f"As the Lead Reasoner, please summarize the best approach to solve the task: {task}."
        ))
        lead_output = cot_agents[0].forward(response_format={
            "thinking": "Your step by step summary of the approach.",
            "answer": "A single letter, A, B, C, or D."
        })
        
        # Final answer from the lead agent
        return lead_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
