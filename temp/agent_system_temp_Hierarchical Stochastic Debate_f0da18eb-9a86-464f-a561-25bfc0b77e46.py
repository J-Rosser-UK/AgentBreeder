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
        # Create a system agent to provide instructions
        system = self.Agent(agent_name='system', temperature=0.8)
    
        # Initialize the primary debate agent
        primary_debate_agent = self.Agent(agent_name='Primary Debate Agent', temperature=0.8)
    
        # Initialize specialized agents for focused feedback
        specialized_agents = [self.Agent(
            agent_name=name,
            temperature=0.7
        ) for name in ['Logic Specialist', 'Content Specialist', 'Context Specialist']]
    
        # Instruction for final decision-making based on all debates and solutions
        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup a single meeting for the debate
        meeting = self.Meeting(meeting_name='hierarchical_stochastic_debate')
    
        # Ensure all agents are part of the meeting
        meeting.agents.extend([system, primary_debate_agent] + specialized_agents)
    
        # Generate response from the primary debate agent
        meeting.chats.append(self.Chat(agent=system, content=f'Please solve the task: {task}'))
        primary_output = primary_debate_agent.forward(response_format={
            'thinking': 'Your step by step reasoning.',
            'response': 'Your final response.',
            'confidence': 'Your confidence score (0-1, as a float).'
        })
        meeting.chats.append(self.Chat(agent=primary_debate_agent, content=primary_output['thinking'] + primary_output['response']))
    
        # Collect feedback from specialized agents
        feedback_outputs = []
        for agent in specialized_agents:
            feedback_output = agent.forward(response_format={
                'feedback': 'Your feedback on the response.',
                'confidence': 'Your confidence score (0-1, as a float).'
            })
            # Ensure confidence is a float
            feedback_outputs.append((feedback_output['feedback'], float(feedback_output['confidence'])))  # Convert confidence to float
            meeting.chats.append(self.Chat(agent=agent, content=feedback_output['feedback']))
    
        # Stochastic decision-making based on feedback confidence scores
        feedbacks, confidences = zip(*feedback_outputs)
        normalized_confidences = np.array(confidences) / np.sum(confidences)
        chosen_feedback = np.random.choice(feedbacks, p=normalized_confidences)
    
        # Final decision based on chosen feedback
        meeting.chats.append(self.Chat(agent=system, content='Based on the feedback, please provide a final answer.'))
        final_output = final_decision_agent.forward(response_format={
            'thinking': 'Your step by step reasoning based on feedback.',
            'answer': 'A single letter, A, B, C or D.'
        })
    
        return final_output['answer']

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
