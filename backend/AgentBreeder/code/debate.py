from backend.AgentBreeder.base import Agent, Meeting, Chat

class Debate:

    def forward(self, task: str) -> str:

        # Create a system agent to provide instructions
        system = Agent(agent_name = 'system', temperature=0.8)

        # Initialize debate agents with different roles and a moderate temperature for varied reasoning
        debate_agents = [Agent(
            agent_name=name,
            temperature=0.8
        ) for name in ['Biology Expert', 'Physics Expert', 'Science Generalist']]

        # Instruction for final decision-making based on all debates and solutions
        final_decision_agent = Agent(agent_name = 'Final Decision Agent',temperature=0.1)
        
        # Setup a single meeting for the debate
        meeting = Meeting(meeting_name="debate")

        # Ensure all agents are part of the meeting
        [meeting.agents.append(agent) for agent in debate_agents]
        meeting.agents.append(system)
        meeting.agents.append(final_decision_agent)

        max_round = 2 # Maximum number of debate rounds

        # Perform debate rounds
        for r in range(max_round):
            for i in range(len(debate_agents)):
                if r == 0 and i == 0:
                    meeting.chats.append(Chat(agent=system, content=f"Please think step by step and then solve the task: {task}"))
                    output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "A single letter, A, B, C or D."})
                    
                else:
                    meeting.chats.append(Chat(agent=system, content=f"Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. Reminder, the task is: {task}"))
                    output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "A single letter, A, B, C or D."})

                meeting.chats.append(Chat(agent=debate_agents[i], content=output["thinking"]+output["response"]))

        # Make the final decision based on all debate results and solutions
        meeting.chats.append(Chat(agent=system, content="Given all the above thinking and answers, reason over them carefully and provide a final answer."))
        output = final_decision_agent.forward(response_format = {"thinking": "Your step by step thinking.", "answer": "A single letter, A, B, C or D."})
       
        return output["answer"]
    

    
    


if __name__ == '__main__':
    
    agent_system = Debate()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)