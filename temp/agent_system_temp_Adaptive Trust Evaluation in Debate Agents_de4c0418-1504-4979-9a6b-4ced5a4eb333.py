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

    def forward(self, session, task: str) -> str:
        # Create a system agent to provide instructions
        system = Agent(session=session, agent_name='system', temperature=0.8)
    
        # Initialize debate agents with memory for learned strategies
        debate_agents = [Agent(session=session, agent_name=name, temperature=0.8) for name in ['Biology Expert', 'Physics Expert', 'Science Generalist']]
    
        # Initialize adversarial agents to challenge the debate agents' responses
        adversarial_agents = [Agent(session=session, agent_name=name, temperature=0.8) for name in ['Critic Biology Expert', 'Critic Physics Expert', 'Critic Generalist']]
    
        # Initialize a final decision agent
        final_decision_agent = Agent(session=session, agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup a single meeting for the debate
        meeting = self.Meeting(meeting_name="adaptive_trust_evaluation_debate")
    
        # Ensure all agents are part of the meeting
        [meeting.agents.append(agent) for agent in debate_agents]
        [meeting.agents.append(agent) for agent in adversarial_agents]
        meeting.agents.append(system)
        meeting.agents.append(final_decision_agent)
    
        max_round = 2  # Maximum number of debate rounds
    
        # Initialize trust scores for debate agents
        trust_scores = {agent.agent_name: 1.0 for agent in debate_agents}
    
        # Perform debate rounds
        for r in range(max_round):
            for i in range(len(debate_agents)):
                meeting.chats.append(self.Chat(agent=system, content=f"Please think step by step and then solve the task: {task}"))
                output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "A single letter, A, B, C, or D."}, task=task)
                meeting.chats.append(self.Chat(agent=debate_agents[i], content=output["thinking"] + output["response"]))
    
                # Adversarial critique
                adversarial_output = adversarial_agents[i].forward(response_format={"critique": "Provide a critique of the previous answer.", "suggestion": "What could be improved?"}, task=task)
                meeting.chats.append(self.Chat(agent=adversarial_agents[i], content=adversarial_output["critique"] + adversarial_output["suggestion"]))
    
                # Update trust scores based on critiques
                score = evaluate_critique(adversarial_output["critique"])  # Assume evaluate_critique is a function that returns a score based on critique quality
                trust_scores[debate_agents[i].agent_name] += score  # Adjust trust score based on critique quality
    
        # Make the final decision based on all debate results and critiques
        meeting.chats.append(self.Chat(agent=system, content="Given all the above thinking and answers, reason over them carefully and provide a final answer considering trust scores."))
        final_output = final_decision_agent.forward(response_format={"thinking": "Your step by step thinking considering critiques and trust scores.", "answer": "A single letter, A, B, C, or D."}, task=task)
        
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
