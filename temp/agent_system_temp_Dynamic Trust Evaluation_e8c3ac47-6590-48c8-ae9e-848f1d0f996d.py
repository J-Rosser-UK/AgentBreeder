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
        
        trust_evaluator = self.Agent(
            agent_name='Trust Evaluator',
            temperature=0.5
        )
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_trust_evaluation")
        meeting.agents.extend([system, cot_agent, critic_agent, trust_evaluator])
        
        N_max = 3  # Maximum number of attempts
        
        # Initialize trust scores
        trust_scores = {"cot_agent": 0, "critic_agent": 0}
        
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
            
            # Update trust scores based on critic feedback
            trust_scores["critic_agent"] += 1 if critic_output["correct"] == "CORRECT" else -1
            
            # Evaluate trustworthiness
            trust_output = trust_evaluator.forward(
                response_format={
                    "cot_answer": output["answer"],
                    "critic_feedback": critic_output["feedback"],
                    "trust_scores": trust_scores
                }
            )
            
            meeting.chats.append(
                self.Chat(
                    agent=trust_evaluator,
                    content=f"Trust evaluation: {trust_output.get('trust_level', 'UNKNOWN')}",
                )
            )
            
            if critic_output["correct"] == "CORRECT" and trust_output.get('trust_level') == 'HIGH':
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
        
        return output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
