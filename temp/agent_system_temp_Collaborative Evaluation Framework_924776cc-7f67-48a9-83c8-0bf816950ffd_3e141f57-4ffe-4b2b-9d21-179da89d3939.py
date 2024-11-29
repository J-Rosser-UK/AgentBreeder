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
        collaborative_eval_agent = self.Agent(agent_name='Collaborative Evaluation Agent', temperature=0.7)
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup shared memory for critiques
        shared_memory_buffer = []  # This will hold expert outputs and critiques
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_eval_meeting")
        meeting.agents.extend([system, collaborative_eval_agent] + list(expert_agents.values()))
        
        # Each expert presents their reasoning
        for expert in expert_agents.values():
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"{expert.agent_name}, please present your reasoning for the task: {task}"
            ))
            expert_output = expert.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C or D."
            })
            meeting.chats.append(self.Chat(
                agent=expert,
                content=expert_output["thinking"]
            ))
            shared_memory_buffer.append(expert_output)  # Collect outputs in shared memory
        
        # Facilitate critiques among experts
        for expert in expert_agents.values():
            meeting.chats.append(self.Chat(
                agent=collaborative_eval_agent,
                content=f"{expert.agent_name}, please critique the reasoning presented by others in the shared memory."
            ))
            critique_output = collaborative_eval_agent.forward(response_format={
                "critiques": "Your critiques of the expert reasoning."
            })
            shared_memory_buffer.append({"critiques": critique_output["critiques"]})  # Store critiques
        
        # Final evaluation based on critiques
        final_answers = [entry['answer'] for entry in shared_memory_buffer if 'answer' in entry]  # Collect answers
        
        # Return the refined answer or the most reliable one if available
        return final_answers[0] if final_answers else "No reliable answer found."  # Ensure reliable fallback.

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
