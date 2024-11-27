from backend.base import Agent, Meeting, Chat

class SelfConsistencyCoT:

    def forward(self, task: str) -> str:
        # Create a system agent to provide instructions
        system = Agent(
            agent_name='system',
            temperature=0.8
        )
        
        # Create multiple CoT agents with higher temperature for varied reasoning
        N = 5  # Number of CoT agents
        cot_agents = [
            Agent(
                agent_name=f'Chain-of-Thought Agent {i}',
                temperature=0.8
            ) for i in range(N)
        ]
        
        # Setup meeting
        meeting = Meeting(meeting_name="self-consistency")
        meeting.agents.extend([system] + cot_agents)
        
        # Collect answers from all agents
        possible_answers = []
        for i in range(N):
            # Add system instruction
            meeting.chats.append(
                Chat(
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
                Chat(
                    agent=cot_agents[i], 
                    content=output["thinking"]
                )
            )
            
            possible_answers.append(output["answer"])

        print(f"Possible answers: {possible_answers}")
        
        # Select the most common answer through majority voting
        from collections import Counter
        
        final_answer = Counter(possible_answers).most_common(1)[0][0]
        return final_answer

if __name__ == '__main__':
    sc_system = SelfConsistencyCoT()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = sc_system.forward(task)
    print(output)