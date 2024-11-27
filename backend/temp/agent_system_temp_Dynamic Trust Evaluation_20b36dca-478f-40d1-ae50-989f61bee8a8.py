import random
import pandas

from base import Agent, Meeting, Chat

class AgentSystem:
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
            agent_name='Trust Agent',
            temperature=0.5
        )
        
        # Setup meeting
        meeting = Meeting(meeting_name="dynamic_trust_reflexion")
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
            
            # Evaluate trustworthiness of the responses
            trust_score_cot = trust_agent.forward(
                response_format={
                    "response": output["answer"],
                    "source": "Chain-of-Thought"
                }
            )
            trust_score_critic = trust_agent.forward(
                response_format={
                    "response": critic_output["feedback"],
                    "source": "Critic"
                }
            )
            
            # Adjust trust scores based on feedback
            if critic_output["correct"] == "CORRECT":
                trust_agent.update_trust_score("Chain-of-Thought", increment=1)
            else:
                trust_agent.update_trust_score("Chain-of-Thought", decrement=1)
                
            # Reflect and refine based on trust scores
            meeting.chats.append(
                Chat(
                    agent=system, 
                    content=f"Given the feedback above, consider where you could go wrong in your latest attempt. Using these insights, try to solve the task better: {task}"
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
