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
        combined_agent = self.Agent(agent_name='Combined Principle and Noise Agent', temperature=0.7)
        cot_agents = [self.Agent(agent_name='CoT Agent 1', temperature=0.8),
                      self.Agent(agent_name='CoT Agent 2', temperature=0.8)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_debate_meeting")
        meeting.agents.extend([system, combined_agent] + cot_agents)
        
        # Get principles and evaluate noise effects in one step
        meeting.chats.append(self.Chat(
            agent=system,
            content="What are the principles involved in solving this task? Please think step by step and explain them. Also, evaluate how noise might affect the reasoning and insights provided."
        ))
        
        combined_output = combined_agent.forward(response_format={
            "thinking": "Your step by step thinking about the principles and noise effects.",
            "principles": "List and explanation of the principles involved.",
            "noise_effect": "Describe how noise impacts the reasoning of the agents."
        })
        
        # Append combined output to the meeting
        meeting.chats.append(self.Chat(
            agent=combined_agent,
            content=combined_output["thinking"] + combined_output["principles"] + combined_output["noise_effect"]
        ))
        
        # Debate phase among CoT agents
        reflections = []
        for cot_agent in cot_agents:
            meeting.chats.append(self.Chat(
                agent=cot_agent,
                content="Based on the principles and noise insights, how would you approach solving the task?"
            ))
            debate_output = cot_agent.forward(response_format={
                "reflection": "Your reflection on the insights provided.",
                "answer": "A single letter, A, B, C, or D."
            })
            reflections.append(debate_output["reflection"])
            meeting.chats.append(self.Chat(
                agent=cot_agent,
                content=debate_output["reflection"]
            ))
        
        # Synthesize insights and solve the task
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Given the principles and insights above, think step by step and solve the task: {task}"
        ))
        
        final_outputs = []
        for cot_agent in cot_agents:
            final_output = cot_agent.forward(response_format={
                "thinking": "Your step by step reasoning considering noise.",
                "answer": "A single letter, A, B, C, or D."
            })
            final_outputs.append(final_output["answer"])
        
        # Determine the final answer based on majority voting
        from collections import Counter
        final_answer = Counter(final_outputs).most_common(1)[0][0]
        
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
