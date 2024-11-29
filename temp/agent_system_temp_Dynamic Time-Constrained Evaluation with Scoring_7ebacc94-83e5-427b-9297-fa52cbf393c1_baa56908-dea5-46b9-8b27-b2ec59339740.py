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
        import time
        # Function to determine time limit based on task complexity
        def determine_time_limit(task):
            # Enhanced complexity evaluation
            complexity_score = 0
            if len(task) < 50:
                complexity_score += 1  # Simple task
            elif len(task) < 100:
                complexity_score += 2  # Moderate task
            else:
                complexity_score += 3  # Complex task
            # Add more complexity factors if needed
            return 3 + complexity_score  # Base time + complexity
    
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        fairness_agent = self.Agent(agent_name='Fairness Agent', temperature=0.7)
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_time_meeting")
        meeting.agents.extend([system, fairness_agent] + list(expert_agents.values()))
        
        expert_outputs = []  # Store expert outputs for evaluation
        time_limit = determine_time_limit(task)  # Dynamic time limit based on task complexity
        
        # Each expert presents their reasoning within the time limit
        for expert in expert_agents.values():
            start_time = time.time()  # Start the timer
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"{expert.agent_name}, please think step by step and present your reasoning for the task: {task}"
            ))
            expert_output = expert.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C or D."
            })
            elapsed_time = time.time() - start_time  # Calculate elapsed time
            if elapsed_time <= time_limit:
                meeting.chats.append(self.Chat(
                    agent=expert,
                    content=expert_output["thinking"]
                ))
                expert_outputs.append(expert_output)  # Collect outputs only if within time limit
            else:
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f"{expert.agent_name} exceeded the time limit of {time_limit} seconds."
                ))
        
        # Ensure at least one output was collected
        if not expert_outputs:
            return "No valid answer could be determined."
    
        # Evaluate arguments for fairness
        evaluations = []
        for output in expert_outputs:
            evaluations.append(output['thinking'])  # Collect expert evaluations
        meeting.chats.append(self.Chat(
            agent=fairness_agent,
            content="Evaluate the following arguments presented by the experts: {evaluations}"
        ))
        
        fairness_output = fairness_agent.forward(response_format={
            "evaluations": "Your evaluations of the arguments."
        })
        
        # Extract the final answer from the fairness evaluation
        final_answer = fairness_output.get("final_answer", None)
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
