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
        # Create a system agent to provide instructions
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create multiple CoT agents for collaboration
        N = 3  # Number of CoT agents
        cot_agents = [
            self.Agent(
                agent_name=f'Chain-of-Thought Agent {i}',
                temperature=0.8
            ) for i in range(N)
        ]
        
        # Setup meeting for collaboration
        meeting = self.Meeting(meeting_name="dynamic_role_assignment_cot")
        meeting.agents.extend([system] + cot_agents)
        
        # Collect reasoning from each agent
        reasoning_outputs = []
        for i in range(N):
            meeting.chats.append(
                self.Chat(
                    agent=system,
                    content=f"Please think step by step and then solve the task: {task}"
                )
            )
            
            # Assign dynamic roles based on task context
            role_assignment = f"Agent {i}, based on your reasoning, you are the Critic for this task."
            meeting.chats.append(
                self.Chat(
                    agent=system,
                    content=role_assignment
                )
            )
            
            # Get reasoning from current CoT agent
            output = cot_agents[i].forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            
            meeting.chats.append(
                self.Chat(
                    agent=cot_agents[i],
                    content=output["thinking"]
                )
            )
            
            reasoning_outputs.append(output)
        
        # Collaborative discussion to refine answers
        for i in range(N):
            discussion_content = "".join([f"Agent {j} thinks: {reasoning_outputs[j]['thinking']}\n" for j in range(N)])
            meeting.chats.append(
                self.Chat(
                    agent=system,
                    content=f"Here are the thoughts from all agents:\n{discussion_content}\nBased on these, please refine your answer."
                )
            )
            refined_output = cot_agents[i].forward(
                response_format={
                    "refined_thinking": "Your refined step by step thinking.",
                    "refined_answer": "A single letter, A, B, C or D."
                }
            )
            meeting.chats.append(
                self.Chat(
                    agent=cot_agents[i],
                    content=refined_output["refined_thinking"]
                )
            )
        
        # Final decision based on refined answers
        final_answers = [refined_output["refined_answer"] for refined_output in reasoning_outputs]
        from collections import Counter
        final_answer = Counter(final_answers).most_common(1)[0][0]
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
