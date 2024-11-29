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
        import time
        # Create system and agent instances
        system = self.Agent(agent_name='system', temperature=0.8)
        cot_agent1 = self.Agent(agent_name='Chain-of-Thought Agent 1', temperature=0.7)
        cot_agent2 = self.Agent(agent_name='Chain-of-Thought Agent 2', temperature=0.7)
        critic_agent1 = self.Agent(agent_name='Critic Agent 1', temperature=0.6)
        critic_agent2 = self.Agent(agent_name='Critic Agent 2', temperature=0.6)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_learning_feedback")
        meeting.agents.extend([system, cot_agent1, cot_agent2, critic_agent1, critic_agent2])
        
        N_max = 3  # Maximum number of attempts
        time_limit = 5  # Time limit in seconds for each CoT agent to respond
        
        # Initial attempts
        meeting.chats.append(
            self.Chat(
                agent=system, 
                content=f"Solve the task: {task}"
            )
        )
        
        def get_output(agent):
            start_time = time.time()
            output = agent.forward(
                response_format={
                    "thinking": "Your reasoning.",
                    "answer": "A single letter, A, B, C, or D."
                }
            )
            time_taken = time.time() - start_time
            if time_taken > time_limit:
                return {"thinking": "Time limit exceeded, please refine your answer.", "answer": output["answer"]}  # Indicate a need for refinement
            return output
        
        outputs = [get_output(agent) for agent in [cot_agent1, cot_agent2]]
        
        for i, output in enumerate(outputs):
            meeting.chats.append(
                self.Chat(
                    agent=meeting.agents[i + 1],  # cot_agent1 or cot_agent2
                    content=output["thinking"]
                )
            )
        
        # Refinement loop
        for i in range(N_max):
            # Get feedback from critics
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content="Review the answers and provide specific feedback on clarity and logic."
                )
            )
            
            critic_outputs = [critic_agent1.forward(response_format={"feedback": "Your feedback on clarity and logic."}), 
                              critic_agent2.forward(response_format={"feedback": "Your feedback on clarity and logic."})]
            
            for j, critic_output in enumerate(critic_outputs):
                meeting.chats.append(
                    self.Chat(
                        agent=meeting.agents[j + 3],  # critic_agent1 or critic_agent2
                        content=critic_output["feedback"]
                    )
                )
            
            # Reflect and refine based on feedback
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content="Based on the feedback, refine your reasoning: {task}"
                )
            )
            
            outputs = [get_output(agent) for agent in [cot_agent1, cot_agent2]]
            
            for i, output in enumerate(outputs):
                meeting.chats.append(
                    self.Chat(
                        agent=meeting.agents[i + 1],  # cot_agent1 or cot_agent2
                        content=output["thinking"]
                    )
                )
        
        # Select the final answer based on consensus of the outputs
        return outputs[0]["answer"] if outputs[0]["answer"] == outputs[1]["answer"] else "Ambiguous"

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
