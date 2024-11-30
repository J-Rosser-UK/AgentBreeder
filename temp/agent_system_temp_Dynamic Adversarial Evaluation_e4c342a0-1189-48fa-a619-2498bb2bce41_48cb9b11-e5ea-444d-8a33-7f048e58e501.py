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
        # Create agents
        system = self.Agent(agent_name='system', temperature=0.8)
        trust_agent = self.Agent(agent_name='Trust Evaluation Agent', temperature=0.7)
        adversarial_agents = [
            self.Agent(agent_name='Adversarial Agent 1', temperature=0.6),
            self.Agent(agent_name='Adversarial Agent 2', temperature=0.6)
        ]
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_adversarial_evaluation_meeting")
        meeting.agents.extend([system, trust_agent] + list(expert_agents.values()) + adversarial_agents)
        
        # Each expert presents reasoning outputs
        expert_outputs = []  # Store expert outputs for evaluation
        for expert in expert_agents.values():
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"{expert.agent_name}, please think step by step and present your reasoning for the task: {task}"
            ))
            expert_output = expert.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C or D."
            })
            meeting.chats.append(self.Chat(
                agent=expert,
                content=expert_output.get("thinking", "No reasoning provided.")
            ))
            # Ensure the output contains a valid answer before appending
            if "answer" in expert_output:
                expert_outputs.append(expert_output)  # Collect valid outputs
    
        # Adversarial agents critique expert outputs and suggest alternatives
        adversarial_outputs = []
        for i, adversary in enumerate(adversarial_agents):
            expert_output = expert_outputs[i]  # Assign each adversary to a specific expert output
            meeting.chats.append(self.Chat(
                agent=adversary,
                content=f"Critique the reasoning and answer provided by the expert: {expert_output['thinking']}"
            ))
            adversarial_output = adversary.forward(response_format={
                "critique": "Your critique of the expert's reasoning.",
                "suggestion": "A single letter, A, B, C or D that you suggest based on your critique."
            })
            meeting.chats.append(self.Chat(
                agent=adversary,
                content=adversarial_output.get("critique", "No critique provided.")
            ))
            adversarial_outputs.append(adversarial_output)  # Collect adversarial critiques
    
        # Voting process among experts
        votes = {output['answer']: 0 for output in expert_outputs if 'answer' in output}
        for output in expert_outputs:
            if 'answer' in output:
                votes[output['answer']] += 1  # Count votes for each answer
    
        # Evaluate arguments for trustworthiness
        meeting.chats.append(self.Chat(
            agent=trust_agent,
            content="Evaluate the reliability of the arguments presented by the experts and determine which responses can be trusted."
        ))
        
        trust_output = trust_agent.forward(response_format={
            "evaluations": "Your evaluations of the arguments based on trustworthiness.",
            "final_answer": "The most trusted answer among the experts based on voting."
        })
        
        # Determine final answer based on votes
        if votes:
            final_answer = max(votes, key=votes.get)
        else:
            final_answer = "A"  # Default answer if no valid votes
        return final_answer  # Return the final answer based on the voting.

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
