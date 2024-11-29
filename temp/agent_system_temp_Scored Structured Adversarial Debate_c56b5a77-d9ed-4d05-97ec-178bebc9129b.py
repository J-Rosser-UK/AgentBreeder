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
    
        # Initialize primary debate agents with different roles and a moderate temperature for varied reasoning
        primary_agents = [self.Agent(
            agent_name=name,
            temperature=0.8
        ) for name in ['Biology Expert', 'Physics Expert', 'Science Generalist']]
    
        # Initialize adversarial agents designed to challenge primary agents
        adversarial_agents = [self.Agent(
            agent_name=name,
            temperature=0.9  # Higher temperature for more aggressive critiques
        ) for name in ['Biology Critic', 'Physics Challenger', 'Science Contrarian']]
    
        # Instruction for final decision-making based on all debates and solutions
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup a single meeting for the debate
        meeting = self.Meeting(meeting_name="scored_structured_adversarial_debate")
    
        # Ensure all agents are part of the meeting
        [meeting.agents.append(agent) for agent in primary_agents]
        [meeting.agents.append(agent) for agent in adversarial_agents]
        meeting.agents.append(system)
        meeting.agents.append(final_decision_agent)
    
        max_round = 2  # Maximum number of debate rounds
    
        # Perform debate rounds
        for r in range(max_round):
            for i in range(len(primary_agents)):
                meeting.chats.append(self.Chat(agent=system, content=f"Please think step by step and then solve the task: {task}"))
                output = primary_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "A single letter, A, B, C, or D."})
                meeting.chats.append(self.Chat(agent=primary_agents[i], content=output["thinking"] + output["response"]))
    
                # Adversarial agents challenge the primary agent's response
                total_score = 0
                for adversary in adversarial_agents:
                    meeting.chats.append(self.Chat(agent=system, content="Now, an adversary will critique the previous response. Please provide your critique."))
                    critique_output = adversary.forward(response_format={"critique": "Your specific critique of the primary agent's answer.", "score": "Score the critique from 1 to 10."})
                    total_score += critique_output["score"]  # Collect scores
                    meeting.chats.append(self.Chat(agent=adversary, content=critique_output["critique"]))
                    meeting.chats.append(self.Chat(agent=primary_agents[i], content="In response to the critique, please clarify your reasoning and provide your answer again."))
                    primary_response = primary_agents[i].forward(response_format={"thinking": "Your refined thinking after critique.", "answer": "A single letter, A, B, C, or D."})
                    meeting.chats.append(self.Chat(agent=primary_agents[i], content=primary_response["thinking"] + primary_response["answer"]))
    
                # Log the average score for this primary agent
                average_score = total_score / len(adversarial_agents)
                meeting.chats.append(self.Chat(agent=system, content=f"Average score for agent {primary_agents[i].agent_name}: {average_score}.")).
    
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
