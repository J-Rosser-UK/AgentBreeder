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
        routing_agent = self.Agent(agent_name='Routing Agent', temperature=0.7)
        competition_agent = self.Agent(agent_name='Competition Agent', temperature=0.7)
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="competitive_collaboration_meeting")
        meeting.agents.extend([system, routing_agent, competition_agent] + list(expert_agents.values()))
        
        # Route the task to the appropriate expert based on complexity
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"Please assess the task complexity and assign the appropriate expert to solve: {task}"
        ))
        
        routing_output = routing_agent.forward(response_format={
            "expert_choice": "Select from: physics, chemistry, biology, or general"
        })
        expert_choice = routing_output["expert_choice"].lower()
        selected_expert = expert_agents.get(expert_choice, expert_agents['general'])  # Default to general if not found
        
        # Expert reasoning with competition focus
        expert_outputs = {}
        for expert in expert_agents.values():
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"{expert.agent_name}, please think step by step and present your reasoning for the task: {task}"
            ))
            expert_output = expert.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C or D."
            })
            meeting.chats.append(self.Chat(
                agent=expert,
                content=expert_output["thinking"]
            ))
            expert_outputs[expert.agent_name] = expert_output
        
        # Evaluate arguments for competition
        meeting.chats.append(self.Chat(
            agent=competition_agent,
            content="Evaluate the reasoning presented by the experts and select the best one."
        ))
        competition_output = competition_agent.forward(response_format={
            "evaluations": "Your structured evaluations of the reasoning from all experts."
        })
        
        # Check if competition_output is structured correctly
        if isinstance(competition_output, dict) and "evaluations" in competition_output:
            # Determine final answer based on competition evaluation
            best_expert_name = max(competition_output["evaluations"], key=competition_output["evaluations"].get)
            final_answer = expert_outputs[best_expert_name].get("answer", None)
        else:
            return "Error: Competition output is not structured correctly."
        
        if final_answer is None:
            return "No valid answer could be determined."
        return final_answer  # Return the final answer based on the evaluation.

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
