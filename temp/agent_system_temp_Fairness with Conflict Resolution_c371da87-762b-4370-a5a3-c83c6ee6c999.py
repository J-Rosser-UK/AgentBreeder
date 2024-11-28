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
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.5)
        
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="fairness_assignment_meeting")
        meeting.agents.extend([system, final_decision_agent] + list(expert_agents.values()))
        
        # Helper function to record expert responses
        def record_expert_response(expert_name, output):
            meeting.chats.append(self.Chat(
                agent=expert_agents[expert_name],
                content=output["thinking"] + output["answer"]
            ))
    
        # Instruct each expert to think independently
        expert_outputs = {}
        for expert_name in expert_agents.keys():
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"{expert_name.capitalize()}, please think step by step and solve the task: {task}"
            ))
            
            expert_output = expert_agents[expert_name].forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C, or D."
            })
            
            # Record each expert's response
            expert_outputs[expert_name] = expert_output
            record_expert_response(expert_name, expert_output)
        
        # Check for conflicts in answers
        answers = [output["answer"] for output in expert_outputs.values()]
        if len(set(answers)) > 1:
            # Conflict resolution: ask for justifications
            for expert_name in expert_agents.keys():
                meeting.chats.append(self.Chat(
                    agent=system,
                    content=f"{expert_name.capitalize()}, please justify your answer: {expert_outputs[expert_name][\"answer\"]}"  # Fixed f-string
                ))
                justification_output = expert_agents[expert_name].forward(response_format={
                    "justification": "Your reasoning for the answer."
                })
                meeting.chats.append(self.Chat(
                    agent=expert_agents[expert_name],
                    content=justification_output["justification"]
                ))
        
        # Final decision-making based on all expert responses
        meeting.chats.append(self.Chat(
            agent=system,
            content="Given all the above thinking and answers, evaluate them carefully and provide a final answer."
        ))
        final_output = final_decision_agent.forward(response_format={
            "thinking": "Your step by step evaluation of all responses.",
            "answer": "A single letter, A, B, C, or D."
        })
        
        return final_output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
