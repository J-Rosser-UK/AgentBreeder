from typing import List, Optional
from datetime import datetime
import uuid
import random
import string
from chat import get_structured_json_response_from_gpt
import json


class Meeting:

    def __init__(self, meeting_name, meeting_timestamp: datetime = datetime.utcnow()):
        self.meeting_id = str(uuid.uuid4())
        self.meeting_name = meeting_name
        self.meeting_timestamp = meeting_timestamp
        self.chats = []
        self.agents = []


class Agent:

    def __init__(
        self,
        agent_name,
        agent_role="",
        agent_goal="None",
        model="gpt-4o-mini",
        temperature=0.7,
        agent_timestamp=datetime.utcnow(),
    ):
        self.agent_id = str(uuid.uuid4())
        characters = string.ascii_letters + string.digits
        random_id = "".join(random.choices(characters, k=4))

        self.agent_name = agent_name + f" {random_id}"

        self.agent_role = agent_role
        self.agent_goal = agent_goal
        self.model = model
        self.temperature = temperature
        self.agent_timestamp = agent_timestamp
        self.chats = []
        self.meetings = []

    def __repr__(self):
        return f"{self.agent_name}"

    @property
    def chat_history(self):

        initial_message = [
            {
                "role": "system",
                "content": f"You are {self.agent_name}. {self.agent_role}. {self.agent_goal}",
            }
        ]

        chats = sorted(
            [chat for meeting in self.meetings for chat in meeting.chats],
            key=lambda x: x.chat_timestamp,
        )

        def to_chat(chat):
            chat_content = chat.content or ""
            if chat.agent.agent_id == self.agent_id:
                role = "assistant"
                content = f"You: {chat_content}"
            elif chat.agent.agent_name == "system":
                role = "system"
                content = f"System: {chat_content}"
            else:
                role = "user"
                content = f"{chat.agent.agent_name}: {chat_content}"
            return {"role": role, "content": content}

        history = initial_message + [to_chat(chat) for chat in chats]

        # print(f"-----{self.agent_name}")
        # for chat in history:
        #     print(json.dumps(chat, indent=4))

        # print("---------------")
        return history

    async def forward(self, response_format: dict):
        # Simulate async response handling
        messages = self.chat_history
        response_json = await get_structured_json_response_from_gpt(
            messages=messages,
            response_format=response_format,
            temperature=self.temperature,
            model=self.model,
        )
        return response_json


class Chat:

    def __init__(
        self,
        agent,
        content=None,
        chat_timestamp=datetime.utcnow(),
    ):
        self.chat_id = str(uuid.uuid4())
        self.agent = agent
        self.content = content
        self.chat_timestamp = chat_timestamp
