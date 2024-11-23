from agent import Agent, Meeting, Chat

class QualityDiversitySystem:
    def forward(self, task: str) -> str:
        # Create agents
        system = Agent(agent_name='system', temperature=0.8)
        cot_agent = Agent(agent_name='Chain-of-Thought Agent', temperature=0.8)
        final_decision_agent = Agent(agent_name='Final Decision Agent', temperature=0.1)
        
        # Setup meeting
        meeting = Meeting(meeting_name="quality_diversity_meeting")
        meeting.agents.extend([system, cot_agent, final_decision_agent])
        
        N_max = 3  # Maximum number of attempts
        
        # Initial attempt
        meeting.chats.append(Chat(
            agent=system,
            content=f"Please think step by step and then solve the task: {task}"
        ))
        
        output = cot_agent.forward(response_format={
            "thinking": "Your step by step thinking.",
            "answer": "A single letter, A, B, C or D."
        })
        
        meeting.chats.append(Chat(
            agent=cot_agent,
            content=output["thinking"] + output["answer"]
        ))
        
        # Generate diverse solutions
        for i in range(N_max):
            meeting.chats.append(Chat(
                agent=system,
                content=f"Given previous attempts, try to come up with another interesting way to solve the task: {task}"
            ))
            
            output = cot_agent.forward(response_format={
                "thinking": "Your step by step thinking with a new approach.",
                "answer": "A single letter, A, B, C or D."
            })
            
            meeting.chats.append(Chat(
                agent=cot_agent,
                content=output["thinking"] + output["answer"]
            ))
        
        # Make final decision
        meeting.chats.append(Chat(
            agent=system,
            content="Given all the above solutions, reason over them carefully and provide a final answer."
        ))
        
        final_output = final_decision_agent.forward(response_format={
            "thinking": "Your step by step thinking comparing all solutions.",
            "answer": "A single letter, A, B, C or D."
        })
        
        return final_output["answer"]


if __name__ == '__main__':
    task = "What is the meaning of life? A: 42 B: 43 C: To live a happy life. D: To do good for others."
    
    qd_system = QualityDiversitySystem()

    print("Quality Diversity System Answer:", qd_system.forward(task))
