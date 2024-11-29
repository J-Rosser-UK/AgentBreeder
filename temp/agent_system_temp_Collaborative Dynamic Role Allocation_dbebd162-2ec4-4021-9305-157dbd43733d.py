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

    class Agent:
        def forward(self, task: str) -> str:
            # Create a system agent to provide instructions
            system = Agent(
                agent_name='system',
                temperature=0.8
            )
            
            # Create multiple agents that can take on dynamic roles
            N = 5  # Number of agents
            agents = [
                Agent(
                    agent_name=f'Agent {i}',
                    temperature=0.7
                ) for i in range(N)
            ]
            
            # Setup meeting
            meeting = Meeting(meeting_name="collaborative_dynamic_role_allocation")
            meeting.agents.extend([system] + agents)
            
            # Evaluate previous performance to assign roles dynamically
            performance_scores = {agent.agent_name: self.evaluate_performance(agent) for agent in agents}
            sorted_agents = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Assign roles based on performance
            thinkers = sorted_agents[:2]  # Top two performers as thinkers
            critic = sorted_agents[2][0]   # Third best as critic
            final_decision_maker = sorted_agents[3][0]  # Fourth best as decision maker
            
            # Add system instruction for the thinkers
            for thinker in thinkers:
                meeting.chats.append(
                    Chat(
                        agent=system,
                        content=f"{thinker[0]}, please think step by step and then solve the task: {task}"
                    )
                )
            
            # Collect responses from thinkers
            responses = []
            for thinker in thinkers:
                thinker_agent = next((agent for agent in agents if agent.agent_name == thinker[0]), None)
                if thinker_agent:
                    output = thinker_agent.forward(
                        response_format={
                            "thinking": "Your step by step thinking.",
                            "answer": "A single letter, A, B, C or D."
                        }
                    )
                    meeting.chats.append(
                        Chat(
                            agent=thinker[0],
                            content=output["thinking"]
                        )
                    )
                    responses.append(output)
            
            # Add system instruction for the critic
            meeting.chats.append(
                Chat(
                    agent=system,
                    content=f"{critic}, please review the answers above and provide feedback."
                )
            )
            
            # Get feedback from critic
            critic_agent = next((agent for agent in agents if agent.agent_name == critic), None)
            if critic_agent:
                critic_output = critic_agent.forward(
                    response_format={
                        "feedback": "Your detailed feedback.",
                        "correct": "Either 'CORRECT' or 'INCORRECT'"
                    }
                )
                meeting.chats.append(
                    Chat(
                        agent=critic,
                        content=critic_output["feedback"]
                    )
                )
            
            # Allow thinkers to revise their answers based on critic feedback
            if critic_output and critic_output["correct"] == "INCORRECT":
                for thinker in thinkers:
                    meeting.chats.append(
                        Chat(
                            agent=system,
                            content=f"{thinker[0]}, based on the feedback, please revise your answer."
                        )
                    )
                    thinker_agent = next((agent for agent in agents if agent.agent_name == thinker[0]), None)
                    if thinker_agent:
                        output = thinker_agent.forward(
                            response_format={
                                "thinking": "Your revised thinking.",
                                "answer": "A single letter, A, B, C or D."
                            }
                        )
                        meeting.chats.append(
                            Chat(
                                agent=thinker[0],
                                content=output["thinking"]
                            )
                        )
            
            # Final decision based on the latest output from the final decision maker
            meeting.chats.append(
                Chat(
                    agent=system,
                    content=f"{final_decision_maker}, please finalize the answer based on the previous feedback and latest thinking."
                )
            )
            final_decision_agent = next((agent for agent in agents if agent.agent_name == final_decision_maker), None)
            if final_decision_agent:
                final_output = final_decision_agent.forward(
                    response_format={
                        "thinking": "Your final considerations.",
                        "answer": "A single letter, A, B, C or D."
                    }
                )
                return final_output["answer"]
            return "Final decision maker not found."

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
