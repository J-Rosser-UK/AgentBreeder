from backend.base import Agent, Meeting, Chat

class RoleAssignmentSystem:
    def forward(self, task: str) -> str:
        # Create agents
        system = Agent(agent_name='system', temperature=0.8)
        routing_agent = Agent(agent_name='Routing Agent', temperature=0.8)
        
        expert_agents = {
            'physics': Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': Agent(agent_name='Biology Expert', temperature=0.8),
            'general': Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = Meeting(meeting_name="role_assignment_meeting")
        meeting.agents.extend([system, routing_agent] + list(expert_agents.values()))
        
        # Route the task
        meeting.chats.append(Chat(
            agent=system,
            content="Given the task, please choose an Expert to answer the question. Choose from: Physics, Chemistry, Biology Expert, or Science Generalist."
        ))
        
        routing_output = routing_agent.forward(response_format={
            "choice": "One of: physics, chemistry, biology, or general"
        })
        
        # Select expert based on routing decision
        expert_choice = routing_output["choice"].lower()
        if expert_choice not in expert_agents:
            expert_choice = 'general'
            
        selected_expert = expert_agents[expert_choice]
        
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
    task = "What is the meaning of life? A: 42 B: 43 C: To live a happy life. D: To do good for others."

    role_system = RoleAssignmentSystem()
    
    print("Role Assignment System Answer:", role_system.forward(task))