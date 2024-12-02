import random
import numpy as np
import pandas

from mocked_base import MockAgent as Agent

from mocked_base import MockMeeting as Meeting

from mocked_base import MockChat as Chat

from sqlalchemy.orm import Session

class AgentSystem:
    def __init__(self, session: Session = None):
        self.Agent = Agent
        self.Meeting = Meeting
        self.Chat = Chat
        self.session = session

    def forward(self, task: str, correct_answer: str) -> str:
        # Create initial generation of agents
        system = self.Agent(agent_name='system', temperature=0.8)
        initial_agents = [self.Agent(agent_name=f'Agent Gen 1-{i}', temperature=0.8) for i in range(5)]
        generations = 3  # Number of generations
        
        for generation in range(generations):
            meeting = self.Meeting(meeting_name=f'cultural_evolution_gen_{generation}')
            meeting.agents.extend([system] + initial_agents)
            outputs = []
            
            # Each agent solves the task
            for agent in initial_agents:
                output = agent.forward(response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }, correct_answer=correct_answer)  # Pass correct_answer here
                outputs.append({"agent_name": agent.agent_name, "thinking": output["thinking"], "answer": output["answer"]})
                meeting.chats.append(self.Chat(
                    agent=agent,
                    content=output["thinking"] + output["answer"]
                ))
            
            # Evaluate outputs and assign scores based on correctness and reasoning quality
            scores = {output["agent_name"]: 0 for output in outputs}
            reasoning_scores = {output["agent_name"]: 0 for output in outputs}
            for output in outputs:
                if output["answer"] == correct_answer:
                    scores[output["agent_name"]] += 1
                reasoning_scores[output["agent_name"]] += len(output["thinking"])  # Example of reasoning quality scoring
            
            # Select best agents to pass on strategies
            best_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]  # Top 3 agents
            
            # Prepare next generation by mutating strategies of best agents
            next_generation = []
            for i in range(5):  # New agents
                if i < len(best_agents):
                    # Clone best agents with slight variations in their strategies
                    best_agent_name = best_agents[i][0]
                    new_agent = self.Agent(agent_name=f'Agent Gen {generation + 1}-{i}', temperature=0.8)
                    new_agent.strategy = f"Mutated strategy from {best_agent_name}"  # Example of strategy sharing
                    next_generation.append(new_agent)
                else:
                    # Create random new agent
                    new_agent = self.Agent(agent_name=f'Agent Gen {generation + 1}-{i}', temperature=0.8)
                    next_generation.append(new_agent)
            
            initial_agents = next_generation  # Update to next generation
        
        # Final decision based on last generation's outputs
        final_meeting = self.Meeting(meeting_name="final_decision")
        final_meeting.agents.extend([system] + initial_agents)
        final_outputs = []
        for agent in initial_agents:
            output = agent.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            }, correct_answer=correct_answer)  # Pass correct_answer here
            final_outputs.append(output["answer"])
        
        # Return the most common answer from the final generation
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
