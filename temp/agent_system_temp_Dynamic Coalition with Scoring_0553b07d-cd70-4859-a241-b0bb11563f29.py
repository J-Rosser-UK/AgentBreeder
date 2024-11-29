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

    def assess_output_quality(output):
        # Placeholder function to assess the quality of the output
        # This can be based on various metrics such as reasoning depth, confidence levels, etc.
        return len(output["thinking"])  # Example metric: length of reasoning as a quality score
    
    
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
    
        # Instruction for final decision-making based on all debates and solutions
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup a single meeting for the debate
        meeting = self.Meeting(meeting_name="dynamic_coalition_debate")
    
        # Ensure all agents are part of the meeting
        [meeting.agents.append(agent) for agent in debate_agents]
        [meeting.agents.append(agent) for agent in adversarial_agents]
        meeting.agents.append(system)
        meeting.agents.append(final_decision_agent)
    
        max_round = 2 # Maximum number of debate rounds
    
        # Perform debate rounds
        for r in range(max_round):
            # Allow agents to form coalitions
            coalitions = []
            scores = {}
            for i in range(len(debate_agents)):
                # Adversarial challenge
                adversarial_output = adversarial_agents[i].forward(response_format={"challenge": "Your challenge to the main agent.", "reasoning": "Your reasoning behind the challenge."})
                meeting.chats.append(self.Chat(agent=adversarial_agents[i], content=adversarial_output["challenge"]))
                meeting.chats.append(self.Chat(agent=adversarial_agents[i], content=adversarial_output["reasoning"]))
                meeting.chats.append(self.Chat(agent=system, content=f"Given the challenge from the adversarial agent, please reconsider your response. The task is: {task}"))
                output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking considering the challenge.", "response": "Your final response.", "answer": "A single letter, A, B, C or D."})
                meeting.chats.append(self.Chat(agent=debate_agents[i], content=output["thinking"] + output["response"]))
    
                # Score the output based on reasoning depth and confidence
                scores[debate_agents[i].agent_name] = assess_output_quality(output)
    
            # Form coalitions based on scores
            sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_agents = [agent[0] for agent in sorted_agents[:2]]  # Top two agents form a coalition
            coalitions.append(top_agents)
    
            # Allow coalition outputs to influence final decision
            for coalition in coalitions:
                coalition_output = [debate_agents[i].forward(response_format={"thinking": "Your coalition reasoning.", "answer": "A single letter, A, B, C or D."}) for i in coalition]
                meeting.chats.append(self.Chat(agent=coalition[0], content="Coalition formed: " + ", ".join(coalition)))
    
        # Make the final decision based on all debate results and coalitions
        meeting.chats.append(self.Chat(agent=system, content="Given all the above thinking and answers, reason over them carefully and provide a final answer."))
        output = final_decision_agent.forward(response_format={"thinking": "Your step by step thinking.", "answer": "A single letter, A, B, C or D."})
        
        return output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
