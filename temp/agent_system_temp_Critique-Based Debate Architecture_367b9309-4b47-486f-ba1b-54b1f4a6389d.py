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
        
        # Initialize two debate agents with specific roles
        debate_agents = [self.Agent(agent_name=name, temperature=0.8) for name in ['Biology Expert', 'Physics Expert']]
        
        # Instruction for final decision-making based on all debates and solutions
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup a single meeting for the debate
        meeting = self.Meeting(meeting_name="critique_debate")
        
        # Ensure all agents are part of the meeting
        [meeting.agents.append(agent) for agent in debate_agents]
        meeting.agents.append(system)
        meeting.agents.append(final_decision_agent)
        
        max_round = 2  # Maximum number of debate rounds
        
        # Perform debate rounds
        for r in range(max_round):
            for i in range(len(debate_agents)):
                if r == 0 and i == 0:
                    meeting.chats.append(self.Chat(agent=system, content=f"Please think step by step and then solve the task: {task}"))
                    output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "A single letter, A, B, C, or D."})
                    # Append combined output as string
                    meeting.chats.append(self.Chat(agent=debate_agents[i], content=f"{output['thinking']} {output['response']}"))
                    
                else:
                    # Critique phase
                    critique_content = f"Critique the previous response of {debate_agents[i-1].agent_name} and provide your improved answer."
                    meeting.chats.append(self.Chat(agent=system, content=critique_content))
                    output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "A single letter, A, B, C, or D."})
                    # Append combined output as string
                    meeting.chats.append(self.Chat(agent=debate_agents[i], content=f"{output['thinking']} {output['response']}"))
    
        # Make the final decision based on all debate results and solutions
        meeting.chats.append(self.Chat(agent=system, content="Given all the above thinking and answers, reason over them carefully and provide a final answer."))
        final_output = final_decision_agent.forward(response_format={"thinking": "Your step by step thinking referencing previous debates.", "answer": "A single letter, A, B, C, or D."})
        # Append the final decision output as string
        meeting.chats.append(self.Chat(agent=final_decision_agent, content=final_output['answer']))
        
        return final_output['answer']

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)