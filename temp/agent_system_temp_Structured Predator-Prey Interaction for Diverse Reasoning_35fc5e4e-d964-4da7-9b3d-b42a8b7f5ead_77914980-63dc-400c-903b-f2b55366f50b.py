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
        
        # Create predator agents that aim for optimal answers
        N_predators = 2  # Number of predator agents
        predators = [
            self.Agent(
                agent_name=f'Predator Agent {i}',
                temperature=0.7
            ) for i in range(N_predators)
        ]
        
        # Create prey agents that generate diverse solutions categorized by reasoning type
        N_prey = 3  # Number of prey agents
        reasoning_types = ['analytical', 'creative', 'heuristic']
        prey_agents = [
            self.Agent(
                agent_name=f'Prey Agent {i}',
                temperature=0.8
            ) for i in range(N_prey)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="structured_predator_prey")
        meeting.agents.extend([system] + predators + prey_agents)
        
        # Prey generate diverse solutions categorized by reasoning type
        prey_answers = {rtype: [] for rtype in reasoning_types}
        for prey in prey_agents:
            for rtype in reasoning_types:
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f"As a {rtype} reasoning agent, your task is to think outside the box. Please consider the question carefully and provide a diverse answer for the task: {task}. Your answer should reflect different perspectives and possible solutions."
                ))
                output = prey.forward(response_format={
                    "thinking": "Your creative reasoning.",
                    "answer": "A single letter, A, B, C, or D."
                })
                meeting.chats.append(self.Chat(
                    agent=prey,
                    content=output["thinking"]
                ))
                prey_answers[rtype].append(output["answer"])
        
        # Predators challenge prey to refine their answers
        refined_answers = []
        for predator in predators:
            for rtype in reasoning_types:
                for prey_answer in prey_answers[rtype]:
                    meeting.chats.append(self.Chat(
                        agent=predator,
                        content=f"Evaluate the following {rtype} answer: {prey_answer}. Is it optimal based on the task requirements? If not, please provide specific suggestions for improvement and a refined answer if possible."
                    ))
                    output = predator.forward(response_format={
                        "feedback": "Your evaluation and specific suggestions."
                    })
                    meeting.chats.append(self.Chat(
                        agent=predator,
                        content=output["feedback"]
                    ))
                    # Check if feedback contains a modified answer
                    if "answer:" in output["feedback"]:
                        refined_answer = output["feedback"].split("answer:")[1].strip()  # Extract the refined answer
                        refined_answers.append(refined_answer)  # Store refined answers only if valid
        
        # Select the best answer from refined answers through majority voting
        from collections import Counter
        if refined_answers:
            final_answer = Counter(refined_answers).most_common(1)[0][0]
        else:
            final_answer = "No valid answer found"
        return final_answer
    

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
