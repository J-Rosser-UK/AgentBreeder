import random
import pandas
import asyncio

from base import Agent, Meeting, Chat


class AgentSystem:

    async def forward(self, task: str) -> str:

        # Create a system agent to provide instructions
        system = Agent(agent_name="system", temperature=0.8)

        # Initialize debate agents with different roles and a moderate temperature for varied reasoning
        debate_agents = [
            Agent(agent_name=name, temperature=0.8)
            for name in ["Biology Expert", "Physics Expert", "Science Generalist"]
        ]

        # Instruction for final decision-making based on all debates and solutions
        final_decision_agent = Agent(agent_name="Final Decision Agent", temperature=0.1)

        # Setup a single meeting for the debate
        meeting = Meeting(meeting_name="debate")

        # Ensure all agents are part of the meeting
        [agent.meetings.append(meeting) for agent in debate_agents]
        system.meetings.append(meeting)
        final_decision_agent.meetings.append(meeting)

        max_round = 2  # Maximum number of debate rounds

        # Perform debate rounds
        for r in range(max_round):
            for i in range(len(debate_agents)):
                if r == 0 and i == 0:
                    meeting.chats.append(
                        Chat(
                            agent=system,
                            content=f"Please think step by step and then solve the task: {task}",
                        )
                    )
                    output = await debate_agents[i].forward(
                        response_format={
                            "thinking": "Your step by step thinking.",
                            "response": "Your final response.",
                            "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING.",
                        }
                    )

                else:
                    meeting.chats.append(
                        Chat(
                            agent=system,
                            content=f"Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. Reminder, the task is: {task}",
                        )
                    )
                    output = await debate_agents[i].forward(
                        response_format={
                            "thinking": "Your step by step thinking.",
                            "response": "Your final response.",
                            "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING.",
                        }
                    )

                meeting.chats.append(
                    Chat(
                        agent=debate_agents[i],
                        content=output["thinking"] + output["response"],
                    )
                )

        # Make the final decision based on all debate results and solutions
        meeting.chats.append(
            Chat(
                agent=system,
                content="Given all the above thinking and answers, reason over them carefully and provide a final answer.",
            )
        )
        output = await final_decision_agent.forward(
            response_format={
                "thinking": "Your step by step thinking.",
                "answer": "Your final answer explicitly in the required answer format as stated in the task. This could be a single letter, a few words or a piece of code. THIS SHOULD NOT BE A SENTENCE AND SHOULD NOT INCLUDE ANY THINKING.",
            }
        )

        return output["answer"]


if __name__ == "__main__":
    from base import initialize_session

    for session in initialize_session():
        agent_system = AgentSystem()
        task = "What should I have for dinner?    A: soup B: burgers C: pizza D: pasta Answer format: a single letter"
        output = asyncio.run(agent_system.forward(task))
        print(output)
