import random
import pandas

from base import Agent, Meeting, Chat

class AgentSystem:
    class TrustAgent(Agent):
        def __init__(self, agent_name, temperature=0.5):
            super().__init__(agent_name=agent_name, temperature=temperature)
            self.trust_scores = {}
    
        def update_score(self, agent_id, adjustment):
            if agent_id not in self.trust_scores:
                self.trust_scores[agent_id] = 0.5  # Initialize trust score to neutral
            self.trust_scores[agent_id] += adjustment
            # Ensure trust scores remain within [0, 1]
            self.trust_scores[agent_id] = max(0, min(1, self.trust_scores[agent_id]))
    
        def get_trust_score(self, agent_id):
            return self.trust_scores.get(agent_id, 0.5)  # Default to neutral if not found
    
    
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
        
        trust_agent = TrustAgent(
            agent_name='Trust Agent',
            temperature=0.5
        )
        
        # Setup meeting
        meeting = Meeting(meeting_name="trust_evaluation")
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
        
        # Refinement loop
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
            
            # Evaluate trust in the critic's feedback
            trust_score = trust_agent.get_trust_score(critic_agent.agent_id)
            
            # Update trust scores based on outcomes
            if critic_output['correct'] == 'INCORRECT':
                trust_agent.update_score(critic_agent.agent_id, -0.1)  # Decrease trust
            else:
                trust_agent.update_score(critic_agent.agent_id, 0.1)  # Increase trust
            
            # Use trust score to decide on refinement
            if trust_score < 0.5:
                # Optionally implement a retry mechanism or gather more feedback
                meeting.chats.append(
                    Chat(
                        agent=system,
                        content="Trust in the critic is low. Please provide additional feedback or reasoning."
                    )
                )
                continue  # Skip this iteration but do not exit the loop
            
            if critic_output["correct"] == "CORRECT":
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
