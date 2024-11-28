import random
import pandas

from base import Agent, Meeting, Chat

from sqlalchemy.orm import Session

class AgentSystem:
    def __init__(self, session: Session):
        self.Agent = Wrapper(Agent, session)
        self.Meeting = Wrapper(Meeting, session)
        self.Chat = Wrapper(Chat, session)
        self.session = session

    def forward(self, task: str) -> str:
        # Create system and specialized agent instances
        system = self.Agent(
            agent_name='system',
            temperature=0.8
        )
        
        reasoning_agent = self.Agent(
            agent_name='Reasoning Agent',
            temperature=0.7
        )
        
        knowledge_agent = self.Agent(
            agent_name='Knowledge Agent',
            temperature=0.7
        )
        
        critique_agent = self.Agent(
            agent_name='Critique Agent',
            temperature=0.6
        )
        
        decision_agent = self.Agent(
            agent_name='Decision Maker',
            temperature=0.5
        )
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="collaborative_decision")
        meeting.agents.extend([system, reasoning_agent, knowledge_agent, critique_agent, decision_agent])
        
        # Each specialized agent provides their perspective
        meeting.chats.append(
            self.Chat(
                agent=system,
                content=f"Please think step by step and then solve the task: {task}"
            )
        )
        
        reasoning_output = reasoning_agent.forward(
            response_format={
                "thinking": "Your reasoning process.",
                "answer": "A single letter, A, B, C or D."
            }
        )
        
        knowledge_output = knowledge_agent.forward(
            response_format={
                "knowledge": "Your relevant knowledge on the subject.",
                "answer": "A single letter, A, B, C or D."
            }
        )
        
        critique_output = critique_agent.forward(
            response_format={
                "critique": "Your critique of the reasoning and knowledge outputs.",
                "feedback": "Any suggestions for improvement."
            }
        )
        
        # Final decision by the decision-making agent
        final_decision = decision_agent.forward(
            response_format={
                "reasoning": reasoning_output,
                "knowledge": knowledge_output,
                "critique": critique_output,
                "final_answer": "Provide the final answer based on all inputs."
            }
        )
        
        return final_decision["final_answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
