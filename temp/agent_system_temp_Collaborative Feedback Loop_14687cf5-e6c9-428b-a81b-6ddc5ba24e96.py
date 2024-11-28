import random
import pandas

from base import Agent, Meeting, Chat

class AgentSystem:
    def forward(self, task: str) -> str:
        # Create agents
        system = Agent(agent_name='system', temperature=0.8)
        routing_agent = Agent(agent_name='Routing Agent', temperature=0.8)
        expert_agents = {
            'physics': Agent(agent_name='Physics Expert', temperature=0.8),
            'chemistry': Agent(agent_name='Chemistry Expert', temperature=0.8),
            'biology': Agent(agent_name='Biology Expert', temperature=0.8),
            'general': Agent(agent_name='Science Generalist', temperature=0.8)
        }
        
        # Setup meeting
        meeting = Meeting(meeting_name="collaborative_feedback_meeting")
        meeting.agents.extend([system, routing_agent] + list(expert_agents.values()))
        
        # Instruct routing agent to manage round-robin discussion
        meeting.chats.append(Chat(
            agent=system,
            content="We will now have a round-robin discussion to solve the task. Each expert will provide their reasoning and answer in turn."
        ))
        
        # Collect answers from all experts in a round-robin fashion
        possible_answers = {}
        reasoning_map = {}
        for expert_name, expert in expert_agents.items():
            meeting.chats.append(Chat(
                agent=system,
                content=f"{expert_name}, please think step by step and then solve the task: {task}"
            ))
            expert_output = expert.forward(response_format={
                "thinking": "Your step by step thinking.",
                "answer": "A single letter, A, B, C or D."
            })
            possible_answers[expert_name] = expert_output["answer"]
            reasoning_map[expert_name] = expert_output["thinking"]
            meeting.chats.append(Chat(
                agent=expert,
                content=expert_output["thinking"] + expert_output["answer"]
            ))
        
        # Allow experts to review reasoning and refine their answers
        for expert_name, expert in expert_agents.items():
            meeting.chats.append(Chat(
                agent=system,
                content=f"{expert_name}, please review the reasoning provided by others: {reasoning_map}. Based on this, refine your answer if necessary."
            ))
            refined_output = expert.forward(response_format={
                "thinking": "Your refined thinking after reviewing others.",
                "answer": "A single letter, A, B, C or D."
            })
            possible_answers[expert_name] = refined_output["answer"]
            meeting.chats.append(Chat(
                agent=expert,
                content=refined_output["thinking"] + refined_output["answer"]
            ))
        
        # Aggregate responses to provide a final answer
        from collections import Counter
        final_answer = Counter(possible_answers.values()).most_common(1)[0][0]
        return final_answer

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
