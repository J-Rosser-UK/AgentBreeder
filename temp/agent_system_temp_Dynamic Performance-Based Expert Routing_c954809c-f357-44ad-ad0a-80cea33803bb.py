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

    class DynamicPerformanceBasedExpertRouting:
        # Class-level variables to maintain state across calls
        selection_counts = {key: 0 for key in ['physics', 'chemistry', 'biology', 'general']}
        performance_scores = {key: 0 for key in ['physics', 'chemistry', 'biology', 'general']}
        decay_factor = 0.9
    
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
            
            # Route the task with performance and fairness consideration
            meeting.chats.append(Chat(
                agent=system,
                content="Given the task, please choose an Expert to answer the question. Choose from: Physics, Chemistry, Biology Expert, or Science Generalist."
            ))
            
            # Dynamic expert selection based on performance and selection counts
            total_selections = sum(self.selection_counts.values())
            expert_choice = max(expert_agents.keys(), key=lambda k: (self.performance_scores[k] * (1 - (self.selection_counts[k] / total_selections))) if total_selections > 0 else 0)
            selected_expert = expert_agents[expert_choice]
            self.selection_counts[expert_choice] += 1
            
            # Get answer from selected expert
            meeting.chats.append(Chat(
                agent=system,
                content=f"Please think step by step and then solve the task: {task}"
            ))
            
            expert_output = selected_expert.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            })
            
            # Update performance scores based on expert output (this is a placeholder for actual evaluation)
            # Here we would typically evaluate the output and adjust scores accordingly
            # For example: self.performance_scores[expert_choice] += evaluate_output(expert_output)
            
            return expert_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
