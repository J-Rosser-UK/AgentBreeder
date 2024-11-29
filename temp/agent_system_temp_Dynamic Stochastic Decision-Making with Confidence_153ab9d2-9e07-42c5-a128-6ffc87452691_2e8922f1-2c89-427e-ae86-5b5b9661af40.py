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
        fairness_agent = self.Agent(agent_name='Fairness Agent', temperature=0.7)
        expert_agents = {
            'physics': self.Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': self.Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': self.Agent(agent_name='Biology Expert', temperature=0.8),
            'general': self.Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = self.Meeting(meeting_name="dynamic_stochastic_decision_meeting")
        meeting.agents.extend([system, fairness_agent] + list(expert_agents.values()))
        
        # Each expert presents their reasoning and assigns a confidence score
        expert_outputs = []  # Store expert outputs for evaluation
        for expert in expert_agents.values():
            meeting.chats.append(self.Chat(
                agent=system,
                content=f"{expert.agent_name}, please think step by step and present your reasoning for the task: {task}"
            ))
            expert_output = expert.forward(response_format={
                "thinking": "Your step by step reasoning.",
                "answer": "A single letter, A, B, C or D.",
                "confidence": "Your confidence level for the answer (0-1)"
            })
            meeting.chats.append(self.Chat(
                agent=expert,
                content=expert_output["thinking"]
            ))
            expert_outputs.append(expert_output)  # Collect outputs
            
            # Validate and convert confidence to float
            confidence = expert_output['confidence']
            if isinstance(confidence, str):
                if confidence == 'high':
                    confidence = 1.0
                elif confidence == 'medium':
                    confidence = 0.5
                elif confidence == 'low':
                    confidence = 0.0
                else:
                    confidence = 0.5  # default to medium if unknown
            else:
                confidence = float(confidence)
            
            # Normalize probabilities
            total_prob = confidence + 0.1 * 4  # 0.1 for each of A, B, C, D
            answer_probs = [confidence / total_prob] + [0.1 / total_prob] * 4  # Normalize to sum to 1
            
            # Stochastic decision-making: select an answer based on normalized probabilities
            selected_answer = np.random.choice([expert_output['answer'], 'A', 'B', 'C', 'D'], p=answer_probs)
            expert_outputs[-1]['selected_answer'] = selected_answer
        
        # Evaluate arguments for fairness
        meeting.chats.append(self.Chat(
            agent=fairness_agent,
            content="Evaluate the arguments presented by the experts and ensure diverse opinions are represented."
        ))
        
        fairness_output = fairness_agent.forward(response_format={
            "evaluations": "Your evaluations of the arguments."
        })
        
        # Extract the final answer from the fairness evaluation
        final_answer = fairness_output.get("final_answer", None)
        if final_answer is None:
            # Fallback to majority voting if no valid answer is provided
            from collections import Counter
            selected_answers = [output['selected_answer'] for output in expert_outputs]
            final_answer = Counter(selected_answers).most_common(1)[0][0]  # Get the most common answer
        return final_answer  # Return the final answer based on the evaluation.

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
