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
        meeting = self.Meeting(meeting_name="collaborative_trust_evaluation")
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
        
        final_answer = None  # Initialize final_answer
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
            
            # Evaluate trust based on critic feedback
            trust_output = trust_agent.forward(
                response_format={
                    "trust_score": "Score of trust based on previous outputs and feedback."
                }
            )
            
            if critic_output["correct"] == "CORRECT":
                final_answer = output["answer"]  # Set final answer if correct
                break
            elif trust_output["trust_score"] < 0.5:  # Low trust score
                # If trust is low, ask for a new answer
                meeting.chats.append(
                    self.Chat(
                        agent=system, 
                        content="The previous answer was deemed unreliable. Please try again: {task}"
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
            else:
                # Collaborative reasoning
                meeting.chats.append(
                    self.Chat(
                        agent=system, 
                        content="Let’s collaboratively reason about the task. Please share your thoughts on the previous answer."
                    )
                )
                # All agents present their reasoning
                reasoning_outputs = []
                for agent in meeting.agents:
                    if agent != system:
                        reasoning_output = agent.forward(
                            response_format={
                                "thinking": "Your step by step reasoning.",
                                "answer": "A single letter, A, B, C or D."
                            }
                        )
                        reasoning_outputs.append(reasoning_output)
                        meeting.chats.append(
                            self.Chat(
                                agent=agent,
                                content=reasoning_output["thinking"]
                            )
                        )
                # Voting mechanism to select the best answer
                if reasoning_outputs:
                    answers = [output["answer"] for output in reasoning_outputs]
                    final_answer = max(set(answers), key=answers.count)  # Majority vote
                    meeting.chats.append(
                        self.Chat(
                            agent=system,
                            content=f"Based on our discussions, the final answer is: {final_answer}"
                        )
                    )
        
        return final_answer if final_answer else "No valid answer found."

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
