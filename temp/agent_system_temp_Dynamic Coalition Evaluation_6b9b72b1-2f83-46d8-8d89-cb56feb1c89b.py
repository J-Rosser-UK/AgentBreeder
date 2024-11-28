import random
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
    
        # Setup a single meeting for coalition discussions
        meeting = self.Meeting(meeting_name="coalition_debate")
    
        # Ensure all agents are part of the meeting
        meeting.agents.extend([system] + debate_agents)
    
        max_round = 2  # Maximum number of debate rounds
    
        # Perform coalition rounds
        for r in range(max_round):
            coalitions = []  # Store coalitions formed in this round
    
            for i in range(len(debate_agents)):
                # Each agent presents its reasoning
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f"Please think step by step and solve the task: {task}"
                ))
                output = debate_agents[i].forward(response_format={
                    "thinking": "Your step by step thinking.",
                    "response": "Your final response.",
                    "answer": "A single letter, A, B, C or D."
                })
    
                # Each agent evaluates the output and decides to form a coalition based on reasoning quality
                if i == 0:
                    coalition = [debate_agents[i]]  # First agent starts a coalition
                else:
                    # Evaluate reasoning quality without inspecting output content directly
                    reasoning_quality = evaluate_reasoning(output["thinking"])
                    if reasoning_quality > 0.5:  # Threshold for forming a coalition
                        coalition.append(debate_agents[i])  # Ally with the first agent
    
                # Store the coalition formed
                if coalition not in coalitions:
                    coalitions.append(coalition)
    
            # Each coalition presents its final answer
            coalition_answers = []
            for coalition in coalitions:
                coalition_response = " + ".join([f"{agent.agent_name}: {output['response']}" for agent in coalition])
                coalition_answers.append(coalition_response)
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f"Coalition formed by {[agent.agent_name for agent in coalition]}: {coalition_response}"
                ))
    
        # Final decision-making based on coalition responses
        meeting.chats.append(self.Chat(agent=system, content="Given all coalition responses, reason over them carefully and provide a final answer."))
        final_output = system.forward(response_format={
            "thinking": "Your final reasoning based on coalition responses.",
            "answer": "A single letter, A, B, C or D."
        })
    
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
