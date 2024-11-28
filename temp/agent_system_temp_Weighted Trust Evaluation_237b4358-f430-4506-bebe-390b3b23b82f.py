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
        system = self.Agent(agent_name='system', temperature=0.8)
        cot_agent = self.Agent(agent_name='Chain-of-Thought Agent', temperature=0.7)
        critic_agent = self.Agent(agent_name='Critic Agent', temperature=0.6)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="weighted_trust_evaluation")
        meeting.agents.extend([system, cot_agent, critic_agent])
        
        # Initialize trust scores
        trust_scores = {agent.agent_name: 1.0 for agent in meeting.agents}  # Trust scores start at 1.0
        
        # Initial attempt
        meeting.chats.append(
            self.Chat(
                agent=system, 
                content=f"Please think step by step and then solve the task: {task}"
            )
        )
        
        answers = []  # Collect answers from all agents
        
        # Get initial answer from the chain-of-thought agent
        output = cot_agent.forward(
            response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            }
        )
        answers.append((output["answer"], trust_scores[cot_agent.agent_name]))
        meeting.chats.append(
            self.Chat(
                agent=cot_agent, 
                content=output["thinking"]
            )
        )
        
        # Collect feedback and evaluate trust
        for _ in range(3):  # Maximum number of attempts
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
            
            # Update trust scores based on feedback
            if critic_output["correct"] == "CORRECT":
                trust_scores[cot_agent.agent_name] += 0.2  # Increase trust
            else:
                trust_scores[cot_agent.agent_name] -= 0.1  # Decrease trust
            
            # Get new answer from the chain-of-thought agent
            output = cot_agent.forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            answers.append((output["answer"], trust_scores[cot_agent.agent_name]))
            meeting.chats.append(
                self.Chat(
                    agent=cot_agent, 
                    content=output["thinking"]
                )
            )
        
        # Final decision based on weighted trust scores
        weighted_answers = {"A": 0, "B": 0, "C": 0, "D": 0}
        for answer, trust in answers:
            weighted_answers[answer] += trust
        final_answer = max(weighted_answers, key=weighted_answers.get)  # Get the answer with the highest score
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
