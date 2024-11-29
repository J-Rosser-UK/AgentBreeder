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
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        cot_agents = [
            self.Agent(agent_name='Chain-of-Thought Agent 1', temperature=0.7),
            self.Agent(agent_name='Chain-of-Thought Agent 2', temperature=0.7),
            self.Agent(agent_name='Chain-of-Thought Agent 3', temperature=0.7),
            self.Agent(agent_name='Chain-of-Thought Agent 4', temperature=0.7)
        ]
        
        critic_agents = [
            self.Agent(agent_name='Critic Agent 1', temperature=0.6),
            self.Agent(agent_name='Critic Agent 2', temperature=0.6)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_reasoning")
        meeting.agents.extend([system] + cot_agents + critic_agents)
        
        N_max = 3  # Maximum number of attempts
        
        # Initial attempt
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
            meeting.chats.append(
                self.Chat(
                    agent=cot_agent, 
                    content=output["thinking"]
                )
            )
            outputs.append(output)
        
        # Collaborative reasoning loop
        for i in range(N_max):
            for j, cot_agent in enumerate(cot_agents):
                # Allow agents to critique each other
                for k, other_agent in enumerate(cot_agents):
                    if j != k:
                        meeting.chats.append(
                            self.Chat(
                                agent=cot_agent,
                                content=f"Critique the reasoning of {other_agent.agent_name}: {outputs[k]["thinking"]}"
                            )
                        )
                        critique_output = cot_agent.forward(
                            response_format={
                                "feedback": "Your feedback on the reasoning.",
                                "improvement": "Suggestions for improvement."
                            }
                        )
                        meeting.chats.append(
                            self.Chat(
                                agent=cot_agent,
                                content=critique_output["feedback"] + " Suggestions: " + critique_output["improvement"]
                            )
                        )
            
            # Get feedback from critics
            all_correct = True
            for critic_agent in critic_agents:
                meeting.chats.append(
                    self.Chat(
                        agent=system, 
                        content="Please review the answers above and criticize where they might be wrong. If you are absolutely sure they are correct, output 'CORRECT'."
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
                if critic_output["correct"] == "INCORRECT":
                    all_correct = False
                    break
            if all_correct:
                break
            
        # Aggregate and return the final answer from the outputs of the last round of CoT agents
        final_answers = [output["answer"] for output in outputs]
        return max(set(final_answers), key=final_answers.count)  # Majority voting for final answer.

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
