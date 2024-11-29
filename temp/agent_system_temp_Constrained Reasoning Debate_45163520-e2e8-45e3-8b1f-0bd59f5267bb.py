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

    class LimitedMemoryAgent:
        def __init__(self, agent, memory_limit=3):
            self.agent = agent
            self.memory_limit = memory_limit
            self.chat_history = []
    
        def forward(self, task, response_format):
            # Add the task to the chat history
            self.chat_history.append(task)
            # Keep only the last `memory_limit` messages
            if len(self.chat_history) > self.memory_limit:
                self.chat_history.pop(0)
            # Call the original agent's forward method with the limited context
            return self.agent.forward(self.chat_history, response_format)
    
    
    def forward(self, task: str) -> str:
        # Create a system agent to provide instructions
        system = self.Agent(agent_name='system', temperature=0.8)
        limited_system = LimitedMemoryAgent(system)
    
        # Initialize main debate agents with limited memory capacity
        debate_agents = [LimitedMemoryAgent(self.Agent(agent_name=name, temperature=0.8)) for name in ['Biology Expert', 'Physics Expert', 'Science Generalist']]
    
        # Initialize adversarial agents with limited reasoning abilities
        adversarial_agents = [LimitedMemoryAgent(self.Agent(agent_name=name, temperature=0.9)) for name in ['Biology Adversary', 'Physics Adversary', 'Generalist Adversary']]
    
        # Instruction for final decision-making based on all debates and solutions
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        limited_final_decision_agent = LimitedMemoryAgent(final_decision_agent)
        
        # Setup a single meeting for the debate
        meeting = self.Meeting(meeting_name="constrained_reasoning_debate")
    
        # Add all agents to the meeting
        for agent in debate_agents + adversarial_agents + [limited_system, limited_final_decision_agent]:
            meeting.agents.append(agent)
    
        max_round = 2  # Maximum number of debate rounds
    
        # Perform debate rounds
        for r in range(max_round):
            for i in range(len(debate_agents)):
                # Adversarial challenge
                adversarial_output = adversarial_agents[i].forward(
                    task=f"Your challenge to the main agent.",
                    response_format={"challenge": "Your challenge to the main agent.", "reasoning": "Your reasoning behind the challenge."}
                )
                meeting.chats.append(self.Chat(agent=adversarial_agents[i], content=adversarial_output["challenge"]))
                meeting.chats.append(self.Chat(agent=adversarial_agents[i], content=adversarial_output["reasoning"]))
                meeting.chats.append(self.Chat(agent=limited_system, content=f"Given the challenge from the adversarial agent, please reconsider your response. The task is: {task}"))
                output = debate_agents[i].forward(
                    task=f"Your step by step thinking considering the challenge.",
                    response_format={"thinking": "Your step by step thinking considering the challenge.", "response": "Your final response.", "answer": "A single letter, A, B, C or D."}
                )
                meeting.chats.append(self.Chat(agent=debate_agents[i], content=output["thinking"] + output["response"]))
    
        # Make the final decision based on all debate results and solutions
        meeting.chats.append(self.Chat(agent=limited_system, content="Given all the above thinking and answers, reason over them carefully and provide a final answer."))
        output = limited_final_decision_agent.forward(
            task="Your step by step thinking.",
            response_format={"thinking": "Your step by step thinking.", "answer": "A single letter, A, B, C or D."}
        )
        
        return output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
