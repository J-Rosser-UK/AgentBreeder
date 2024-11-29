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
        # Create a system agent to provide instructions
        system = self.Agent(agent_name='system', temperature=0.8)
        routing_agent = self.Agent(agent_name='Routing Agent', temperature=0.8)
    
        # Initialize debate agents with different roles and map them for quick access
        debate_agents = {
            'Biology Expert': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'Physics Expert': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'Science Generalist': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
    
        # Instruction for final decision-making based on all debates and solutions
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup a single meeting for the debate
        meeting = self.Meeting(meeting_name="expert_routing_debate")
    
        # Ensure all agents are part of the meeting
        meeting.agents.append(system)
        meeting.agents.append(routing_agent)
        meeting.agents.extend(debate_agents.values())
    
        # Route the task to select relevant agents
        meeting.chats.append(self.Chat(
            agent=system,
            content="Given the task, please choose the most relevant experts to answer the question."
        ))
        routing_output = routing_agent.forward(response_format={
            "selected_experts": "List of selected expert agents"
        })
        selected_experts = routing_output["selected_experts"].split(",")
    
        # Perform debate rounds with selected experts
        for expert_name in selected_experts:
            selected_agent = debate_agents.get(expert_name.strip())
            if selected_agent:
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f"Please think step by step and then solve the task: {task}"
                ))
                output = selected_agent.forward(response_format={
                    "thinking": "Your step by step thinking.",
                    "response": "Your final response.",
                    "answer": "A single letter, A, B, C, or D."
                })
    
                meeting.chats.append(self.Chat(
                    agent=selected_agent,
                    content=output["thinking"] + output["response"]
                ))
    
        # Make the final decision based on all debate results and solutions
        meeting.chats.append(self.Chat(agent=system, content="Given all the above thinking and answers, reason over them carefully and provide a final answer."))
        output = final_decision_agent.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C, or D."
        })
        
        return output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
