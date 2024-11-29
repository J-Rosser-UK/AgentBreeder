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
        system = self.Agent(agent_name='system', temperature=0.8)
    
        # Initialize main debate agents with different roles
        debate_agents = [self.Agent(
            agent_name=name,
            temperature=0.8
        ) for name in ['Biology Expert', 'Physics Expert', 'Science Generalist']]
    
        # Initialize adversarial agents to challenge the debate agents
        adversarial_agents = [self.Agent(
            agent_name=name,
            temperature=0.9
        ) for name in ['Biology Adversary', 'Physics Adversary', 'Generalist Adversary']]
    
        # Initialize shared resource
        shared_resource = 10  # Total available tokens
        resource_spent = {agent.agent_name: 0 for agent in debate_agents}  # Track resource spent by each agent
    
        # Instruction for final decision-making based on all debates and solutions
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup a single meeting for the debate
        meeting = self.Meeting(meeting_name="dynamic_resource_allocation_debate")
    
        # Ensure all agents are part of the meeting
        [meeting.agents.append(agent) for agent in debate_agents]
        [meeting.agents.append(agent) for agent in adversarial_agents]
        meeting.agents.append(system)
        meeting.agents.append(final_decision_agent)
    
        max_round = 2 # Maximum number of debate rounds
    
        # Perform debate rounds
        for r in range(max_round):
            for i in range(len(debate_agents)):
                # Adversarial challenge
                adversarial_output = adversarial_agents[i].forward(response_format={"challenge": "Your challenge to the main agent.", "reasoning": "Your reasoning behind the challenge."})
                meeting.chats.append(self.Chat(agent=adversarial_agents[i], content=adversarial_output["challenge"]))
                meeting.chats.append(self.Chat(agent=adversarial_agents[i], content=adversarial_output["reasoning"]))
                meeting.chats.append(self.Chat(agent=system, content=f"Given the challenge from the adversarial agent, please reconsider your response. The task is: {task}"))
                
                # Each debate agent proposes how much of the shared resource to spend
                resource_to_spend = min(3, shared_resource)  # Example logic to spend up to 3 tokens
                justification = f"I believe spending {resource_to_spend} tokens is necessary to enhance my response due to the complexity of the task."
                resource_spent[debate_agents[i].agent_name] += resource_to_spend
                shared_resource -= resource_to_spend
    
                output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking considering the challenge.", "response": "Your final response.", "answer": "A single letter, A, B, C, or D.", "justification": justification})
                meeting.chats.append(self.Chat(agent=debate_agents[i], content=output["thinking"] + output["response"] + f" (Spent: {resource_to_spend}, Justification: {justification})"))
    
        # Make the final decision based on all debate results and solutions
        meeting.chats.append(self.Chat(agent=system, content="Given all the above thinking and answers, reason over them carefully and provide a final answer."))
        output = final_decision_agent.forward(response_format={"thinking": "Your step by step thinking.", "answer": "A single letter, A, B, C, or D."})
        
        return output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
