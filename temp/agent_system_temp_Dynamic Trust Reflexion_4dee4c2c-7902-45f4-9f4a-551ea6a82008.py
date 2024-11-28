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
        # Create system and agent instances
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        cot_agent = self.Agent(
            agent_name='Chain-of-Thought Agent',
            temperature=0.7
        )
        
        critic_agent = self.Agent(
            agent_name='Critic Agent',
            temperature=0.6
        )
        
        trust_agent = self.Agent(
            agent_name='Trust Evaluation Agent',
            temperature=0.5
        )
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_trust_reflexion")
        meeting.agents.extend([system, cot_agent, critic_agent, trust_agent])
        
        N_max = 3  # Maximum number of attempts
        
        # Initial attempt
        meeting.chats.append(
            self.Chat(
                agent=system, 
                content=f"Please think step by step and then solve the task: {task}"
            )
        )
        
        output = cot_agent.forward(
            response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            }
        )
        
        meeting.chats.append(
            self.Chat(
                agent=cot_agent, 
                content=output["thinking"]
            )
        )
        
        # Refinement loop
        for i in range(N_max):
            # Get feedback from critic
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content="Please review the answer above and criticize where it might be wrong. If you are absolutely sure it is correct, output 'CORRECT'."
                )
            )
            
            critic_output = critic_agent.forward(
                response_format={
                    "feedback": "Your detailed feedback.",
                    "correct": "Either 'CORRECT' or 'INCORRECT'"
                }
            )
            
            meeting.chats.append(
                self.Chat(
                    agent=critic_agent, 
                    content=critic_output["feedback"]
                )
            )
            
            # Trust evaluation based on critic's feedback
            trust_output = trust_agent.forward(
                response_format={
                    "trust_score_cot": "Trust score for Chain-of-Thought Agent",
                    "trust_score_critic": "Trust score for Critic Agent"
                }
            )
            
            # Ensure trust scores are numeric
            cot_weight = self.map_trust_score_to_numeric(trust_output["trust_score_cot"])
            critic_weight = self.map_trust_score_to_numeric(trust_output["trust_score_critic"])
            
            # Use trust scores to determine weight of responses
            total_weight = cot_weight + critic_weight
            if total_weight == 0:
                total_weight = 1  # Prevent division by zero
            
            # Calculate weighted answer based on trust scores
            weighted_answer = (cot_weight * output["answer"] + critic_weight * critic_output["feedback"]) / total_weight
            
            if critic_output["correct"] == "CORRECT":
                break
            
            # Reflect and refine
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Given the feedback above, carefully consider where you could go wrong in your latest attempt. Using these insights, try to solve the task better: {task}"
                )
            )
            
            output = cot_agent.forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            
            meeting.chats.append(
                self.Chat(
                    agent=cot_agent, 
                    content=output["thinking"]
                )
            )
        
        return weighted_answer
    
    # Define the trust score mapping inside the class to ensure proper context
        def map_trust_score_to_numeric(self, trust_score: str) -> float:
            mapping = {
                'high': 1.0,
                'medium': 0.5,
                'low': 0.0
            }
            return mapping.get(trust_score.lower(), 0.0)  # Default to 0.0 if not found

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
