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
        
        # Initialize main debate agents and adversarial agents
        debate_agents = [self.Agent(agent_name=name, temperature=0.8) for name in ['Biology Expert', 'Physics Expert', 'Science Generalist']]
        adversarial_agents = [self.Agent(agent_name='Adversarial Agent', temperature=0.9) for _ in range(2)]  # Two adversarial agents
        
        # Instruction for final decision-making based on all debates and solutions
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup a single meeting for the debate
        meeting = self.Meeting(meeting_name="dynamic_adversarial_feedback_debate")
        
        # Ensure all agents are part of the meeting
        meeting.agents.extend(debate_agents + adversarial_agents + [system, final_decision_agent])
        
        max_round = 2  # Maximum number of debate rounds
        
        # Perform debate rounds
        for r in range(max_round):
            for i in range(len(debate_agents)):
                # Main agent responds to the task
                if r == 0 and i == 0:
                    meeting.chats.append(self.Chat(agent=system, content=f"Please think step by step and then solve the task: {task}"))
                else:
                    meeting.chats.append(self.Chat(agent=system, content=f"Given solutions from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. Reminder, the task is: {task}"))
                
                output = debate_agents[i].forward(response_format={
                    "thinking": "Your step by step thinking.",
                    "response": "Your final response.",
                    "answer": "A single letter, A, B, C, or D."
                })
                meeting.chats.append(self.Chat(agent=debate_agents[i], content=output["thinking"] + output["response"]))
                
                # Adversarial agents present their challenges
                for adv_agent in adversarial_agents:
                    adversarial_output = adv_agent.forward(response_format={
                        "challenge": "Present a conflicting argument or misleading information.",
                        "response": "Your response to the main agent's answer."
                    })
                    meeting.chats.append(self.Chat(agent=adv_agent, content=adversarial_output["challenge"] + adversarial_output["response"]))
                    
                    # The main agent must respond to the adversarial challenge
                    meeting.chats.append(self.Chat(agent=system, content=f"How do you respond to the challenge: {adversarial_output[\"challenge\"]}?"))
                    response_to_challenge = debate_agents[i].forward(response_format={
                        "defense": "Your defense against the challenge.",
                        "answer": "A single letter, A, B, C, or D."
                    })
                    meeting.chats.append(self.Chat(agent=debate_agents[i], content=response_to_challenge["defense"] + response_to_challenge["answer"]))
                    
                    # Collect feedback from the adversarial agent
                    feedback = adv_agent.forward(response_format={
                        "feedback": "Provide feedback on the main agent's response."
                    })
                    meeting.chats.append(self.Chat(agent=adv_agent, content=feedback["feedback"]))
                    
                    # Allow the main agent to refine their answer based on feedback
                    refined_output = debate_agents[i].forward(response_format={
                        "thinking": "Your refined step by step thinking.",
                        "answer": "A single letter, A, B, C, or D."
                    })
                    meeting.chats.append(self.Chat(agent=debate_agents[i], content=refined_output["thinking"] + refined_output["answer"]))
        
        # Make the final decision based on all debate results and solutions
        output = final_decision_agent.forward(response_format={
            "thinking": "Your step by step thinking over the debate results.",
            "answer": "A single letter, A, B, C, or D."
        })
        
        return output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
