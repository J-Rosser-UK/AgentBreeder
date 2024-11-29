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
        # Create system and agent instances
        system = self.Agent(agent_name='system', temperature=0.8)
        N_generations = 3  # Number of generations
        agents = []
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="refined_cultural_evolution_meeting")
        meeting.agents.append(system)
        
        for generation in range(N_generations):
            # Create agents for this generation
            generation_agents = [self.Agent(agent_name=f'Agent Gen {generation} - {i}', temperature=0.7) for i in range(3)]
            agents.extend(generation_agents)
            meeting.agents.extend(generation_agents)
            
            # Each agent solves the task
            successful_strategies = []
            for agent in generation_agents:
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f"Agent {agent.agent_name}, please solve the task: {task} step by step."
                ))
                output = agent.forward(response_format={
                    "thinking": "Your step by step reasoning.",
                    "answer": "A, B, C, or D."
                })
                meeting.chats.append(self.Chat(
                    agent=agent,
                    content=f"Answer: {output['answer']} with reasoning: {output['thinking']}"
                ))
                
                # Critique the answer
                meeting.chats.append(self.Chat(
                    agent=agent,
                    content=f"Critique your answer: {output['answer']} and reasoning: {output['thinking']}"
                ))
                critique_output = agent.forward(response_format={
                    "critique": "Your critique of the previous answer."
                })
                meeting.chats.append(self.Chat(
                    agent=agent,
                    content=critique_output["critique"]
                ))
                
                # Store successful strategies based on critique
                if output['answer'] in ['A', 'B', 'C', 'D']:  # Assuming valid answers
                    successful_strategies.append((output['answer'], output['thinking']))
    
            # Share only successful strategies with the next generation
            if generation < N_generations - 1:
                meeting.chats.append(self.Chat(
                    agent=system,
                    content="Please share your successful strategies with the next generation."
                ))
                for strategy in successful_strategies:
                    meeting.chats.append(self.Chat(
                        agent=system,
                        content=f"Successful Strategy: {strategy[0]} with reasoning: {strategy[1]}"
                    ))
    
        # Final decision from the last generation
        meeting.chats.append(self.Chat(
            agent=system,
            content="Review all successful strategies from previous generations and provide a final answer."
        ))
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        final_output = final_decision_agent.forward(response_format={
            "thinking": "Your reasoning based on all successful strategies.",
            "answer": "A, B, C, or D."
        })
        
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
