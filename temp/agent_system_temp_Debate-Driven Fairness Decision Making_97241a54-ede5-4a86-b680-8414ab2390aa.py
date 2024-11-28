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
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create multiple CoT agents to explore the task
        N = 3  # Number of CoT agents
        cot_agents = [
            self.Agent(
                agent_name=f'Chain-of-Thought Agent {i}',
                temperature=0.8
            ) for i in range(N)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="debate_fairness_decision")
        meeting.agents.extend([system] + cot_agents)
        
        # Collect initial answers from all agents
        agent_responses = []
        for i in range(N):
            # Add system instruction
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Please think step by step and then solve the task: {task}"
                )
            )
            
            # Get response from current COT agent
            output = cot_agents[i].forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            
            # Record the agent's response
            meeting.chats.append(
                self.Chat(
                    agent=cot_agents[i], 
                    content=output["thinking"]
                )
            )
            
            agent_responses.append(output)
    
        # Debate phase: Allow agents to challenge each other's answers
        for i in range(N):
            meeting.chats.append(
                self.Chat(
                    agent=system,
                    content="Now, please respond to the answers of other agents and provide your reasoning for your answer."
                )
            )
            
            debate_output = cot_agents[i].forward(
                response_format={
                    "debate_thinking": "Your rebuttal reasoning.",
                    "debate_answer": "A single letter, A, B, C or D."
                }
            )
            
            meeting.chats.append(
                self.Chat(
                    agent=cot_agents[i], 
                    content=debate_output["debate_thinking"]
                )
            )
    
        # Final decision based on fairness
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.5)
        meeting.chats.append(
            self.Chat(
                agent=system,
                content="Given all the responses and rebuttals from the agents, please evaluate the answers based on fairness and thorough reasoning."
            )
        )
        
        # Analyze responses and select the best answer
        final_decision_output = final_decision_agent.forward(response_format={
            "evaluated_answers": "Analyze all answers and provide the best one based on fairness."
        })
        
        return final_decision_output["evaluated_answers"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
