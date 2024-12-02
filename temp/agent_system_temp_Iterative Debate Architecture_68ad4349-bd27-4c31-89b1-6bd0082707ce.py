import random
import numpy as np
import pandas

from mocked_base import MockAgent as Agent

from mocked_base import MockMeeting as Meeting

from mocked_base import MockChat as Chat

from sqlalchemy.orm import Session

class AgentSystem:
    def __init__(self, session: Session = None):
        self.Agent = Agent
        self.Meeting = Meeting
        self.Chat = Chat
        self.session = session

    def forward(self, task: str) -> str:\n    # Create a system agent to provide instructions\n    system = self.Agent(agent_name='system', temperature=0.8)\n\n    # Initialize two debate agents with different roles\n    debate_agents = [self.Agent(\n        agent_name=name,\n        temperature=0.8\n    ) for name in ['Biology Expert', 'Physics Expert']]\n\n    # Instruction for final decision-making based on all debates and solutions\n    final_decision_agent = self.Agent(agent_name='Final Decision Agent', temperature=0.1)\n    \n    # Setup a single meeting for the debate\n    meeting = self.Meeting(meeting_name="debate")\n\n    # Ensure all agents are part of the meeting\n    [meeting.agents.append(agent) for agent in debate_agents]\n    meeting.agents.append(system)\n    meeting.agents.append(final_decision_agent)\n\n    # First round: each agent presents their reasoning and answer\n    outputs = []\n    for i in range(len(debate_agents)):\n        meeting.chats.append(self.Chat(agent=system, content=f"Please think step by step and then solve the task: {task}"))\n        output = debate_agents[i].forward(response_format={"thinking": "Your step by step thinking.", "response": "Your final response.", "answer": "A single letter, A, B, C or D."})\n        meeting.chats.append(self.Chat(agent=debate_agents[i], content=output["thinking"] + output["response"]))\n        outputs.append(output)\n\n    # Feedback phase: agents critique each other's responses\n    for i in range(len(debate_agents)):\n        feedback = f"Critique the response of {debate_agents[1-i].agent_name}: {outputs[1-i]['response']}"\n        meeting.chats.append(self.Chat(agent=system, content=feedback))\n        critique_output = debate_agents[i].forward(response_format={"thinking": "Your critique.", "refinement": "Your refined answer."})\n        meeting.chats.append(self.Chat(agent=debate_agents[i], content=critique_output["thinking"] + critique_output["refinement"]))\n\n    # Final decision based on all debate results and refined answers\n    meeting.chats.append(self.Chat(agent=system, content="Given all the above thinking and answers, reason over them carefully and provide a final answer."))\n    output = final_decision_agent.forward(response_format={"thinking": "Your step by step thinking.", "answer": "A single letter, A, B, C or D."})\n    \n    return output["answer"]

if __name__ == '__main__':
    from base import initialize_session
    session, Base = initialize_session
    agent_system = AgentSystem()
    task = "What is the meaning of life? A: 42 B: 43 C: To life a happy life. D: To do good for others."
    output = agent_system.forward(task)
    print(output)
