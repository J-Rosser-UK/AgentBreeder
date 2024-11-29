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
    
        # Initialize debate agents with different roles and a moderate temperature for varied reasoning
        debate_agents = [self.Agent(
            agent_name=name,
            temperature=0.8
        ) for name in ['Biology Expert', 'Physics Expert', 'Science Generalist']]
    
        # Create a Critic Agent to evaluate responses
        critic_agent = self.Agent(agent_name='Critic Agent', temperature=0.6)
    
        # Instruction for final decision-making based on all debates and solutions
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup a single meeting for the debate
        meeting = self.Meeting(meeting_name="debate_with_critique")
    
        # Ensure all agents are part of the meeting
        [meeting.agents.append(agent) for agent in debate_agents]
        meeting.agents.append(system)
        meeting.agents.append(critic_agent)
        meeting.agents.append(final_decision_agent)
    
        max_round = 2 # Maximum number of debate rounds
    
        # Perform debate rounds
        for r in range(max_round):
            for i in range(len(debate_agents)):
                if r == 0 and i == 0:
                    meeting.chats.append(self.Chat(agent=system, content=f"Please think step by step and then solve the task: {task}"))
                    output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "A single letter, A, B, C, or D."})
                else:
                    meeting.chats.append(self.Chat(agent=system, content=f"Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. Reminder, the task is: {task}"))
                    output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "A single letter, A, B, C, or D."})
    
                meeting.chats.append(self.Chat(agent=debate_agents[i], content=output["thinking"] + output["response"]))
    
                # Critique the response after each agent
                meeting.chats.append(self.Chat(agent=critic_agent, content=f"Critique the response from {debate_agents[i].agent_name}: {output['response']}"))
                critique_output = critic_agent.forward(response_format={"feedback": "Your feedback on the response."})
                meeting.chats.append(self.Chat(agent=critic_agent, content=critique_output["feedback"]))
    
                # Allow debate agents to revise their responses based on the critique
                meeting.chats.append(self.Chat(agent=debate_agents[i], content=f"Based on the critique, please revise your response."))
                revised_output = debate_agents[i].forward(response_format={"thinking": "Your revised step by step thinking.", "response": "Your revised final response.", "answer": "A single letter, A, B, C, or D."})
                meeting.chats.append(self.Chat(agent=debate_agents[i], content=revised_output["response"]))
    
        # Make the final decision based on all debate results and solutions
        meeting.chats.append(self.Chat(agent=system, content="Given all the above thinking and critiques, reason over them carefully and provide a final answer."))
        output = final_decision_agent.forward(response_format={"thinking": "Your step by step thinking based on critiques.", "answer": "A single letter, A, B, C, or D."})
        
        return output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
