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
        # Create a system agent to provide instructions
        system = self.Agent(agent_name='system', temperature=0.8)
    
        # Initialize debate agents with different roles
        debate_agents = [self.Agent(
            agent_name=name,
            temperature=0.8
        ) for name in ['Biology Expert', 'Physics Expert', 'Science Generalist']]
    
        # Setup a single meeting for the debate
        meeting = self.Meeting(meeting_name="multi_role_adaptive_debate")
        meeting.agents.extend([system] + debate_agents)
    
        max_round = 2  # Maximum number of debate rounds
    
        # Store performance metrics for dynamic allocation
        performance_metrics = {agent.agent_name: {'correct': 0, 'total': 0, 'reasoning_quality': []} for agent in debate_agents}
    
        # Perform debate rounds
        for r in range(max_round):
            # Reset performance metrics for the round
            for agent in debate_agents:
                performance_metrics[agent.agent_name]['correct'] = 0
                performance_metrics[agent.agent_name]['total'] = 0
                performance_metrics[agent.agent_name]['reasoning_quality'] = []
    
            for agent in debate_agents:
                # Each agent presents their reasoning
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f"Please solve the task: {task}"
                ))
                output = agent.forward(response_format={
                    "thinking": "Your step by step thinking.",
                    "response": "Your final response.",
                    "answer": "A single letter, A, B, C, or D.",
                    "reasoning_quality": "Rate your reasoning quality from 1 to 5."
                })
                meeting.chats.append(self.Chat(agent=agent, content=output["thinking"] + output["response"]))
    
                # Evaluate performance and store metrics
                performance_metrics[agent.agent_name]['total'] += 1
                if output["answer"] == 'A':  # Assuming 'A' is the correct answer for this task
                    performance_metrics[agent.agent_name]['correct'] += 1
                performance_metrics[agent.agent_name]['reasoning_quality'].append(int(output["reasoning_quality"]))
    
            # Dynamic role allocation based on performance
            for agent in debate_agents:
                correctness_rate = performance_metrics[agent.agent_name]['correct'] / performance_metrics[agent.agent_name]['total'] if performance_metrics[agent.agent_name]['total'] > 0 else 0
                avg_reasoning_quality = sum(performance_metrics[agent.agent_name]['reasoning_quality']) / len(performance_metrics[agent.agent_name]['reasoning_quality']) if performance_metrics[agent.agent_name]['reasoning_quality'] else 0
                # Assign roles based on performance metrics
                agent.role = "Lead" if correctness_rate > 0.8 and avg_reasoning_quality > 3 else "Supporter" if correctness_rate > 0.5 else "Challenger"
    
        # Final decision based on all debate results and solutions
        meeting.chats.append(self.Chat(agent=system, content="Given all the above thinking and answers, reason over them carefully and provide a final answer."))
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        final_output = final_decision_agent.forward(response_format={"thinking": "Your step by step thinking based on the debate results.", "answer": "A single letter, A, B, C, or D."})
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
