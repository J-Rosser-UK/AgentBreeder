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
        # Create system and agent instances
        system = self.Agent(agent_name='system', temperature=0.8)
        cot_agent1 = self.Agent(agent_name='Chain-of-Thought Agent 1', temperature=0.7)
        cot_agent2 = self.Agent(agent_name='Chain-of-Thought Agent 2', temperature=0.7)
        critic_agent = self.Agent(agent_name='Critic Agent', temperature=0.6)
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_resource_allocation")
        meeting.agents.extend([system, cot_agent1, cot_agent2, critic_agent])
        
        # Initialize performance tracking
        performance_scores = {cot_agent1.agent_id: 0, cot_agent2.agent_id: 0}
        token_limit = 50  # Initial token limit for both agents
        
        # Initial attempts
        meeting.chats.append(
            self.Chat(
                agent=system, 
                content=f"Please think step by step and then solve the task: {task} (You have a limit of {token_limit} tokens for your reasoning)",
            )
        )
        
        outputs = []
        for agent in [cot_agent1, cot_agent2]:
            output = agent.forward(
                response_format={
                    "thinking": "Your step by step thinking (50 tokens max).",
                    "answer": "A single letter, A, B, C, or D."
                }
            )
            # Check token usage
            thinking_tokens = len(output['thinking'].split())
            if thinking_tokens > token_limit:
                output['thinking'] = ' '.join(output['thinking'].split()[:token_limit])  # Enforce token limit
            outputs.append(output)
            meeting.chats.append(
                self.Chat(
                    agent=agent, 
                    content=output["thinking"]
                )
            )
        
        # Critic reviews the outputs
        meeting.chats.append(
            self.Chat(
                agent=system, 
                content="Please review the answers above and provide detailed feedback including a score and suggestions for improvement."
            )
        )
        critic_output = critic_agent.forward(
            response_format={
                "feedback": "Your detailed feedback.",
                "score": "Score out of 10",  # Ensure this is an integer
                "suggestions": "Specific suggestions for improvement"
            }
        )
        
        meeting.chats.append(
            self.Chat(
                agent=critic_agent, 
                content=critic_output["feedback"]
            )
        )
        
        # Update performance scores based on critic feedback
        for i, agent in enumerate([cot_agent1, cot_agent2]):
            performance_scores[agent.agent_id] += int(critic_output["score"])  # Ensure this is treated as an integer
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content="Given the feedback and suggestions, refine your reasoning and try again (50 tokens max): {task}"
                )
            )
            refined_output = agent.forward(
                response_format={
                    "thinking": "Your refined step by step thinking (50 tokens max).",
                    "answer": "A single letter, A, B, C, or D."
                }
            )
            meeting.chats.append(
                self.Chat(
                    agent=agent, 
                    content=refined_output["thinking"]
                )
            )
        
        # Determine final answer based on outputs
        answers = [output["answer"] for output in outputs]
        if answers[0] == answers[1]:
            return answers[0]
        else:
            # Implement a more sophisticated resolution mechanism for ambiguity
            return max(set(answers), key=answers.count)  # Majority vote for the final answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
