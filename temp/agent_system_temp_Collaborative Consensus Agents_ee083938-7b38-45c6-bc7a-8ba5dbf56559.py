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
        import time
        from sqlalchemy.exc import OperationalError
        
        # Create a system agent to provide instructions
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create collaborative agents with varied temperatures to generate outputs
        collaborative_agents = [
            self.Agent(agent_name=f'Collaborative Agent {i}', temperature=0.5 + 0.1 * i)
            for i in range(3)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_consensus")
        meeting.agents.extend([system] + collaborative_agents)
        
        # Instruct collaborative agents to generate outputs
        outputs = []
        for agent in collaborative_agents:
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"Please think step by step and solve the task: {task}"
            ))
            
            output = agent.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C, or D."
            })
            
            # Retry mechanism for database insert
            retries = 3
            for attempt in range(retries):
                try:
                    meeting.chats.append(self.Chat(
                        agent=agent,
                        content=output["thinking"] + output["answer"]
                    ))
                    break  # Exit loop if successful
                except OperationalError:
                    if attempt < retries - 1:
                        time.sleep(1)  # Wait before retrying
                    else:
                        raise  # Raise error if all retries failed
            
            outputs.append(output)
        
        # Facilitate a structured collaborative review of outputs
        meeting.chats.append(self.Chat(
            agent=system,
            content="Now, please review the outputs provided by your peers and critique them constructively."
        ))
        
        for agent in collaborative_agents:
            agent_feedback = agent.forward(response_format={
                "outputs": outputs,
                "feedback": "Your feedback on the outputs."
            })["feedback"]
            meeting.chats.append(self.Chat(
                agent=agent,
                content=agent_feedback
            ))
        
        # Collect suggestions for improvements
        suggestions = []
        for agent in collaborative_agents:
            suggestion = agent.forward(response_format={
                "outputs": outputs,
                "suggestion": "Your suggestion for improvement."
            })["suggestion"]
            meeting.chats.append(self.Chat(
                agent=agent,
                content=suggestion
            ))
            suggestions.append(suggestion)
        
        # Final consensus on the answer based on quality of reasoning
        final_answers = [output["answer"] for output in outputs]
        final_best_answer = max(set(final_answers), key=final_answers.count)  # Majority voting
        return final_best_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
