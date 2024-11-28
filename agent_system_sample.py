from sqlalchemy.orm import Session
from base import Agent, Meeting, Chat, Wrapper



class AgentSystem:
    def __init__(self, session: Session):
        self.Agent = Wrapper(Agent, session)
        self.Meeting = Wrapper(Meeting, session)
        self.Chat = Wrapper(Chat, session)
        self.session = session

    def forward(self, task: str) -> str:
        system = self.Agent(agent_name='system', temperature=0.8)

        debate_agents = [self.Agent(
            agent_name=name,
            temperature=0.8
        ) for name in ['Biology Expert', 'Physics Expert', 'Science Generalist']]

        final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)

        meeting = self.Meeting(meeting_name="debate")
        [meeting.agents.append(agent) for agent in debate_agents]
        meeting.agents.append(system)
        meeting.agents.append(final_decision_agent)

        max_round = 2
        for r in range(max_round):
            for i in range(len(debate_agents)):
                if r == 0 and i == 0:
                    meeting.chats.append(self.Chat(
                        agent=system,
                        content=f"Please think step by step and solve the task: {task}"
                    ))
                    output = debate_agents[i].forward(response_format={
                        "thinking": "Your step-by-step thinking.",
                        "response": "Your final response.",
                        "answer": "A single letter, A, B, C or D."
                    })
                else:
                    meeting.chats.append(self.Chat(
                        agent=system,
                        content=f"Given solutions from others, consider their opinions as advice. Task: {task}"
                    ))
                    output = debate_agents[i].forward(response_format={
                        "thinking": "Your step-by-step thinking.",
                        "response": "Your final response.",
                        "answer": "A single letter, A, B, C or D."
                    })

                meeting.chats.append(self.Chat(agent=debate_agents[i], content=output["thinking"] + output["response"]))

        meeting.chats.append(self.Chat(agent=system, content="Given all the above, provide a final answer."))
        output = final_decision_agent.forward(response_format={
            "thinking": "Your step-by-step thinking.",
            "answer": "A single letter, A, B, C or D."
        })

        return output["answer"]



if __name__ == '__main__':
    from base import initialize_session
    import threading

    from concurrent.futures import ThreadPoolExecutor
    def run_task():
        session, Base = initialize_session()
        debate_instance = AgentSystem(session)
        task = "What is the meaning of life? A: 42 B: 43 C: To live a happy life. D: To do good for others."
        return debate_instance.forward(task)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_task) for _ in range(10)]
        for future in futures:
            print(future.result())
