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

    class Agent:
        def __init__(self, agent_name, temperature=0.5):
            self.agent_name = agent_name
            self.temperature = temperature
            self.memory = []  # Initialize memory to store successful strategies
    
        def adapt_strategy(self, critique):
            # Implement logic to adapt the agent's strategy based on the critique
            pass
    
        def get_expected_answer(self, task):
            # Logic to retrieve the expected answer based on the task
            return "A"  # Placeholder implementation
    
    
    def forward(self, task: str) -> str:
        # Create a system agent to provide instructions
        system = Agent(agent_name='system', temperature=0.8)
    
        # Initialize debate agents with memory for learned strategies
        debate_agents = [Agent(agent_name=name, temperature=0.8) for name in ['Biology Expert', 'Physics Expert', 'Science Generalist']]
    
        # Initialize adversarial agents to challenge the debate agents' responses
        adversarial_agents = [Agent(agent_name=name, temperature=0.8) for name in ['Critic Biology Expert', 'Critic Physics Expert', 'Critic Generalist']]
    
        # Initialize a final decision agent
        final_decision_agent = Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup a single meeting for the debate
        meeting = self.Meeting(meeting_name="dynamic_strategy_evolution_debate")
    
        # Ensure all agents are part of the meeting
        [meeting.agents.append(agent) for agent in debate_agents]
        [meeting.agents.append(agent) for agent in adversarial_agents]
        meeting.agents.append(system)
        meeting.agents.append(final_decision_agent)
    
        max_round = 2  # Maximum number of debate rounds
    
        # Perform debate rounds
        for r in range(max_round):
            for i in range(len(debate_agents)):
                meeting.chats.append(self.Chat(agent=system, content=f"Please think step by step and then solve the task: {task}"))
                output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "A single letter, A, B, C, or D."})
                meeting.chats.append(self.Chat(agent=debate_agents[i], content=output["thinking"] + output["response"]))
    
                # Adversarial critique
                adversarial_output = adversarial_agents[i].forward(response_format={"critique": "Provide a critique of the previous answer.", "suggestion": "What could be improved?"})
                meeting.chats.append(self.Chat(agent=adversarial_agents[i], content=adversarial_output["critique"] + adversarial_output["suggestion"]))
    
                # Evaluate and evolve strategies
                expected_answer = system.get_expected_answer(task)  # Retrieve the expected answer
                if output["answer"] == expected_answer:  # Check if the answer is correct
                    debate_agents[i].memory.append(output["thinking"])  # Store successful strategies
                else:
                    debate_agents[i].adapt_strategy(adversarial_output["critique"])  # Adapt strategy based on critique
    
        # Make the final decision based on all debate results and critiques
        meeting.chats.append(self.Chat(agent=system, content="Given all the above thinking and answers, reason over them carefully and provide a final answer."))
        final_output = final_decision_agent.forward(response_format={"thinking": "Your step by step thinking considering critiques.", "answer": "A single letter, A, B, C, or D."})
        
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
