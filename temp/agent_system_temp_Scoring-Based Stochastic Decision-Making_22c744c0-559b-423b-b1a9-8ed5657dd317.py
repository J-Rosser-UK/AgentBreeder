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

    def evaluate_reasoning(reasoning: str) -> int:
        # A simple scoring mechanism based on keyword presence
        keywords = ['correct', 'valid', 'supported', 'clear', 'logical']
        score = sum(1 for keyword in keywords if keyword in reasoning.lower())
        return score
    
    
    def forward(self, task: str) -> str:
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        routing_agent = self.Agent(agent_name='Routing Agent', temperature=0.7)
        stochastic_agent = self.Agent(agent_name='Stochastic Decision Agent', temperature=0.5)
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="scoring_based_stochastic_decision_meeting")
        meeting.agents.extend([system, routing_agent, stochastic_agent] + list(expert_agents.values()))
        
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
        
        # Expert reasoning
        meeting.chats.append(self.Chat(
            agent=system,
            content=f"{selected_expert.agent_name}, please think step by step and present your reasoning for the task: {task}"
        ))
        expert_output = selected_expert.forward(response_format={
            "thinking": "Your step by step reasoning.",
            "answer": "A single letter, A, B, C or D."
        })
        meeting.chats.append(self.Chat(
            agent=selected_expert,
            content=expert_output["thinking"]
        ))
        
        # Collect reasoning outputs for stochastic selection
        reasoning_outputs = [expert_output["answer"]]
        scores = [1]  # Assume initial score of 1 for the first expert
        for expert in expert_agents.values():
            if expert != selected_expert:
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f"{expert.agent_name}, please provide your reasoning for the task: {task}"
                ))
                output = expert.forward(response_format={
                    "thinking": "Your step by step reasoning.",
                    "answer": "A single letter, A, B, C or D."
                })
                reasoning_outputs.append(output["answer"])
                scores.append(evaluate_reasoning(output["thinking"]))  # Evaluate the reasoning quality
    
        # Stochastic decision-making to choose an answer based on scores
        total_score = sum(scores)
        probabilities = [score / total_score for score in scores]  # Normalize scores to probabilities
        selected_answer_index = np.random.choice(len(reasoning_outputs), p=probabilities)  # Select index based on probabilities
        return reasoning_outputs[selected_answer_index]  # Return the randomly selected answer based on probabilities.

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
