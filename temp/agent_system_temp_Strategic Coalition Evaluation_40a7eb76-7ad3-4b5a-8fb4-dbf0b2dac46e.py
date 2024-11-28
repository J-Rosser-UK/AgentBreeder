import random
import pandas

from base import Agent, Meeting, Chat

class AgentSystem:
    def forward(self, task: str) -> str:
        # Create agents
        system = Agent(agent_name='system', temperature=0.8)
        expert_agents = {
            'physics': Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': Agent(agent_name='Biology Expert', temperature=0.8),
            'general': Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = Meeting(meeting_name="strategic_coalition_meeting")
        meeting.agents.extend([system] + list(expert_agents.values()))
        
        # Agents evaluate each other
        evaluations = {}
        for agent_name, agent in expert_agents.items():
            evaluation_output = agent.forward(response_format={
                "evaluation": "Rate your peers on a scale from 1 to 5 based on expertise.",
                "reasoning": "Explain your evaluation."
            })
            evaluations[agent_name] = evaluation_output
            meeting.chats.append(Chat(
                agent=agent,
                content=f"I evaluate {agent_name} as {evaluation_output['evaluation']} because: {evaluation_output['reasoning']}"
            ))
        
        # Form coalitions based on evaluations
        coalitions = []
        for agent_name, evaluation in evaluations.items():
            if evaluation['evaluation'] >= 3:  # Threshold for joining a coalition
                coalitions.append(agent_name)
                meeting.chats.append(Chat(
                    agent=expert_agents[agent_name],
                    content=f"I will join the coalition as I rated myself and others positively."
                ))
        
        # If no coalitions were formed, fallback to general expert
        selected_expert = expert_agents['general'] if not coalitions else expert_agents[coalitions[0]]
        
        # Get answer from selected expert
        meeting.chats.append(Chat(
            agent=system,
            content=f"Please think step by step and then solve the task: {task}"
        ))
        
        expert_output = selected_expert.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        
        return expert_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
