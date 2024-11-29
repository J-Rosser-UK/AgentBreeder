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
        cot_agents = [self.Agent(agent_name=f'Chain-of-Thought Agent {i+1}', temperature=0.7) for i in range(4)]
        critics = [self.Agent(agent_name=f'Critic Agent {i+1}', temperature=0.6) for i in range(4)]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="structured_reflexion")
        meeting.agents.extend([system] + cot_agents + critics)
        
        N_max = 3  # Maximum number of attempts
        
        # Initial attempts
        meeting.chats.append(
            self.Chat(
                agent=system, 
                content=f"Please think step by step and then solve the task: {task}"
            )
        )
        
        outputs = []
        for cot_agent in cot_agents:
            output = cot_agent.forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            outputs.append(output)
            meeting.chats.append(
                self.Chat(
                    agent=cot_agent, 
                    content=output["thinking"]
                )
            )
        
        # Refinement loop
        for i in range(N_max):
            # Get feedback from critics
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content="Please review the answers above and provide feedback."
                )
            )
            critic_outputs = []
            for j, critic in enumerate(critics):
                critic_output = critic.forward(
                    response_format={
                        "feedback": f"Your feedback on the output from {cot_agents[j].agent_name}."
                    }
                )
                critic_outputs.append(critic_output)
                meeting.chats.append(
                    self.Chat(
                        agent=critic, 
                        content=critic_output["feedback"]
                    )
                )
            
            # Reflect and refine based on feedback
            for j, cot_agent in enumerate(cot_agents):
                meeting.chats.append(
                    self.Chat(
                        agent=system, 
                        content="Given the feedback, refine your reasoning and try again: {task}"
                    )
                )
                output = cot_agent.forward(
                    response_format={
                        "thinking": "Your refined step by step thinking.",
                        "answer": "A single letter, A, B, C or D."
                    }
                )
                outputs[j] = output  # Update the output
                meeting.chats.append(
                    self.Chat(
                        agent=cot_agent, 
                        content=output["thinking"]
                    )
                )
        
        # Select the final answer based on consensus of the outputs
        answers = [output["answer"] for output in outputs]
        from collections import Counter
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common()  # Get the most common answer
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            return {"consensus": None, "status": "Ambiguous"}
        else:
            return {"consensus": most_common[0][0], "status": "Consensus"}

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
