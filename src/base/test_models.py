import sys

sys.path.append("src")
from models import Agent, Meeting, Chat


class AgentSystem:

    def forward(self, task: str) -> str:
        system = Agent(agent_name="system", temperature=0.8)

        debate_agents = [
            Agent(agent_name=name, temperature=0.8)
            for name in ["Biology Expert", "Physics Expert", "Science Generalist"]
        ]

        final_decision_agent = Agent(agent_name="Final Decision Agent", temperature=0.1)

        meeting = Meeting(meeting_name="debate")
        [agent.meetings.append(meeting) for agent in debate_agents]
        system.meetings.append(meeting)
        final_decision_agent.meetings.append(meeting)

        max_round = 2
        for r in range(max_round):
            for i in range(len(debate_agents)):
                if r == 0 and i == 0:
                    meeting.chats.append(
                        Chat(
                            agent=system,
                            content=f"Please think step by step and solve the task: {task}",
                        )
                    )
                    output = debate_agents[i].forward(
                        response_format={
                            "thinking": "Your step-by-step thinking.",
                            "response": "Your final response.",
                            "answer": "A single letter, A, B, C or D.",
                        }
                    )
                else:
                    meeting.chats.append(
                        Chat(
                            agent=system,
                            content=f"Given solutions from others, consider their opinions as advice. Task: {task}",
                        )
                    )
                    output = debate_agents[i].forward(
                        response_format={
                            "thinking": "Your step-by-step thinking.",
                            "response": "Your final response.",
                            "answer": "A single letter, A, B, C or D.",
                        }
                    )

                meeting.chats.append(
                    Chat(
                        agent=debate_agents[i],
                        content=output["thinking"] + output["response"],
                    )
                )

        meeting.chats.append(
            Chat(agent=system, content="Given all the above, provide a final answer.")
        )
        output = final_decision_agent.forward(
            response_format={
                "thinking": "Your step-by-step thinking.",
                "answer": "A single letter, A, B, C or D.",
            }
        )

        return output["answer"]


if __name__ == "__main__":

    from concurrent.futures import ThreadPoolExecutor

    def run_task():

        debate_instance = AgentSystem()
        task = "What is the meaning of life? A: 42 B: 43 C: To live a happy life. D: To do good for others."
        return debate_instance.forward(task)

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_task) for _ in range(1)]
        for future in futures:
            print(future.result())
