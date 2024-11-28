import random
import pandas

from base import Agent, Meeting, Chat

class AgentSystem:
    class TrustScoreManager:
        def __init__(self):
            self.scores = {}
    
        def update_score(self, agent_name, feedback):
            if agent_name not in self.scores:
                self.scores[agent_name] = 0
            # Simple scoring logic based on feedback
            if feedback == 'CORRECT':
                self.scores[agent_name] += 1
            else:
                self.scores[agent_name] -= 1
    
        def get_score(self, agent_name):
            return self.scores.get(agent_name, 0)
    
    
    def forward(self, task: str) -> str:
        # Create system and agent instances
        system = Agent(
            agent_name='system',
            temperature=0.8
        )
        
        cot_agent = Agent(
            agent_name='Chain-of-Thought Agent',
            temperature=0.7
        )
        
        critic_agent = Agent(
            agent_name='Critic Agent',
            temperature=0.6
        )
        
        trust_agent = Agent(
            agent_name='Trust Evaluation Agent',
            temperature=0.5
        )
        
        trust_manager = TrustScoreManager()  # Initialize trust score manager
        
        # Setup meeting
        meeting = Meeting(meeting_name="dynamic_trust_evaluation")
        meeting.agents.extend([system, cot_agent, critic_agent, trust_agent])
        
        N_max = 3  # Maximum number of attempts
        
        # Initial attempt
        meeting.chats.append(
            Chat(
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
            Chat(
                agent=cot_agent, 
                content=output["thinking"]
            )
        )
        
        # Trust evaluation loop
        for i in range(N_max):
            # Get feedback from critic
            meeting.chats.append(
                Chat(
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
                Chat(
                    agent=critic_agent, 
                    content=critic_output["feedback"]
                )
            )
            
            # Update trust score based on critic's feedback
            trust_manager.update_score(cot_agent.agent_name, critic_output["correct"])
            
            # Evaluate trust based on updated scores
            trust_score = trust_manager.get_score(cot_agent.agent_name)
            trust_level = "Low" if trust_score < 0 else "High" if trust_score > 0 else "Medium"
            meeting.chats.append(
                Chat(
                    agent=trust_agent, 
                    content=f"Trust level evaluated as: {trust_level}"
                )
            )
            
            if critic_output["correct"] == "CORRECT" or trust_level == "High":
                break
            
            # Reflect and refine
            meeting.chats.append(
                Chat(
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
                Chat(
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
