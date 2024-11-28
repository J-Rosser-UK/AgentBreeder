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
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create multiple CoT agents for varied reasoning
        N = 3  # Number of CoT agents
        cot_agents = [
            self.Agent(
                agent_name=f'Chain-of-Thought Agent {i}',
                temperature=0.8
            ) for i in range(N)
        ]
        
        # Create adversarial agents to challenge the CoT agents
        M = 2  # Number of adversarial agents
        adversarial_agents = [
            self.Agent(
                agent_name=f'Adversarial Agent {i}',
                temperature=0.7
            ) for i in range(M)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="adversarial_challenge")
        meeting.agents.extend([system] + cot_agents + adversarial_agents)
        
        # Collect answers from all CoT agents
        possible_answers = []
        for i in range(N):
            # Add system instruction
            meeting.chats.append(
                self.Chat(
                    agent=system, 
                    content=f"Please think step by step and then solve the task: {task}"
                )
            )
            
            # Get response from current COT agent
            output = cot_agents[i].forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
            
            # Record the agent's response
            meeting.chats.append(
                self.Chat(
                    agent=cot_agents[i], 
                    content=output["thinking"]
                )
            )
            
            possible_answers.append(output["answer"])
        
        # Now let adversarial agents challenge the answers
        challenge_results = []
        for j in range(M):
            meeting.chats.append(
                self.Chat(
                    agent=system,
                    content="Please review the answers from the CoT agents and provide counterarguments or alternative solutions."
                )
            )
            
            # Get response from adversarial agent
            adversarial_output = adversarial_agents[j].forward(
                response_format={
                    "challenge": "Your counterarguments or alternative solutions."
                }
            )
            
            # Record the adversarial agent's response
            meeting.chats.append(
                self.Chat(
                    agent=adversarial_agents[j], 
                    content=adversarial_output["challenge"]
                )
            )
            challenge_results.append(adversarial_output["challenge"])
        
        # Synthesize final answer based on CoT answers and adversarial challenges
        final_answer = self.synthesize_final_answer(possible_answers, challenge_results)
        return final_answer
    
    # Additional method to synthesize the final answer based on scores
    def synthesize_final_answer(self, possible_answers, challenge_results):
        scores = {answer: 0 for answer in possible_answers}
        for challenge in challenge_results:
            # Evaluate challenges against possible answers and update scores
            for answer in possible_answers:
                # Simple scoring logic: increase score if challenge is relevant to answer
                if challenge in answer:  # Placeholder for actual evaluation logic
                    scores[answer] += 1
        # Select the answer with the highest score
        final_answer = max(scores, key=scores.get)
        return final_answer
    

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
