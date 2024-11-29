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
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Create a system agent to provide instructions
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create multiple CoT agents with higher temperature for varied reasoning
        N = 3  # Number of CoT agents
        cot_agents = [
            self.Agent(
                agent_name=f'Chain-of-Thought Agent {i}',
                temperature=0.8
            ) for i in range(N)
        ]
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="parallel_time_constrained_decision")
        meeting.agents.extend([system] + cot_agents)
        
        possible_answers = []
        time_limit = 5  # Time limit in seconds for each agent to respond
        
        def get_agent_response(agent):
            # Add system instruction
            chat_entry = self.Chat(
                agent=system, 
                content=f"Please think step by step and then solve the task: {task}"
            )
            # Store chat entry temporarily
            return chat_entry, agent.forward(
                response_format={
                    "thinking": "Your step by step thinking.",
                    "answer": "A single letter, A, B, C or D."
                }
            )
    
        with ThreadPoolExecutor(max_workers=N) as executor:
            future_to_agent = {executor.submit(get_agent_response, agent): agent for agent in cot_agents}
            for future in as_completed(future_to_agent):
                chat_entry, output = future.result()
                # Append chat entry to meeting chats
                meeting.chats.append(chat_entry)
                # Check response time
                elapsed_time = time.time() - start_time
                if elapsed_time <= time_limit:
                    possible_answers.append(output["answer"])
                else:
                    possible_answers.append('C')  # Default answer if time limit exceeded
    
        # Commit the meeting changes after processing all responses
        # Assuming a commit method is available in the session context
        meeting.commit()  # Ensure we commit after processing answers
    
        # Aggregate responses using weighted voting based on confidence levels
        from collections import Counter
        final_answer = Counter(possible_answers).most_common(1)[0][0]
        return final_answer
    

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
