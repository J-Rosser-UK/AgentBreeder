import datetime
import uuid
import random
import string
from chat import get_structured_json_response_from_gpt
import httpx

client = httpx.AsyncClient()


class Chat:
    def __init__(self, agent, content):
        self.chat_id = str(uuid.uuid4())
        self.agent = agent
        self.content = content
        self.chat_timestamp = (
            datetime.datetime.utcnow()
        )  # Added parentheses to call the method

    def __repr__(self):
        return f"\n\n{self.agent}: {self.content}"


class Meeting:
    def __init__(self, meeting_name):
        self.meeting_id = str(uuid.uuid4())
        self.meeting_name = meeting_name
        self.meeting_timestamp = (
            datetime.datetime.utcnow()
        )  # Added parentheses to call the method
        self.chats = []


class Agent:
    def __init__(self, agent_name, model="gpt-4o-mini", temperature=0.5):
        self.agent_id = str(uuid.uuid4())
        self.agent_name = agent_name
        self.meetings = []
        self.model = model
        self.temperature = temperature
        self.agent_timestamp = (
            datetime.datetime.utcnow()
        )  # Added parentheses to call the method
        characters = (
            string.ascii_letters + string.digits
        )  # includes both upper/lower case letters and numbers
        random_id = "".join(random.choices(characters, k=4))
        self.agent_name = agent_name + " " + random_id

    def __repr__(self):
        return f"{self.agent_name}"

    @property
    def chat_history(self):
        meetings = self.meetings
        chats = []
        for meeting in meetings:
            chats.extend(meeting.chats)

        print(chats)

        # order chats by timestamp
        chats = sorted(chats, key=lambda x: x.chat_timestamp)

        # Convert into the format [{role: agent, content: chat_content}]
        def to_chat(chat):
            chat_content: str = chat.content if chat.content else ""

            if chat.agent.agent_id == self.agent_id:
                role = "assistant"
                content = "You: " + chat_content
            elif chat.agent.agent_name == "system":
                role = "system"
                content = "System: " + chat_content
            else:
                role = "user"
                content = chat.agent.agent_name + ": " + chat_content

            return {"role": role, "content": content}

        chats = [to_chat(chat) for chat in chats]
        return chats

    def forward(self, response_format) -> dict:
        messages = self.chat_history

        # send an asynchttp get request to wikipedia
        response_json = client.get(
            "https://en.wikipedia.org/wiki/Python_(programming_language)"
        )

        # response_json = get_structured_json_response_from_gpt(
        #     messages=messages, response_format=response_format, temperature=0.5
        # )
        print(response_json)

        return response_json
