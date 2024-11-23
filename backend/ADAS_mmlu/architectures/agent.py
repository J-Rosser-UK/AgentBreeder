from sqlalchemy import Column, String, Float, JSON, UUID, ForeignKey, DateTime
import datetime
from sqlalchemy.orm import declarative_base, relationship
import uuid
from chat import get_structured_json_response_from_gpt
from icecream import ic
import random
import string


Base = declarative_base()


class CustomBase(Base):
    __abstract__ = True

    def __init__(self, **kwargs):
        super().__init__()
        
        # First, set all default values for columns
        for column in self.__table__.columns:
            if column.default is not None:
                if column.default.is_scalar:
                    setattr(self, column.name, column.default.arg)
                elif column.type.__class__ == UUID:
                    setattr(self, column.name, uuid.uuid4())
                elif column.type.__class__ == DateTime:
                    setattr(self, column.name, datetime.utcnow())

        # Then override with any provided values
        for attr, value in kwargs.items():
            setattr(self, attr, value)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

class CustomColumn(Column):
    def __init__(self, *args, label=None, **kwargs):
        self.label = label
        super().__init__(*args, **kwargs)
    

class Chat(CustomBase):
    __tablename__ = 'chat'

    chat_id = CustomColumn(UUID, primary_key=True, default=lambda: uuid.uuid4(), label="The chat's unique identifier (UUID).")
    agent_id = CustomColumn(String, ForeignKey('agent.agent_id'), nullable=False, label="The role of the chat.")
    meeting_id = CustomColumn(UUID, ForeignKey('meeting.meeting_id'), nullable=False, label="The meeting's unique identifier (UUID).")
    content = CustomColumn(String, nullable=False, label="The content of the chat.")
    chat_timestamp = CustomColumn(DateTime, default=datetime.datetime.now(), label="The timestamp of the chat.")

    # Relationships
    agent = relationship("Agent", back_populates="chats")
    meeting = relationship("Meeting", back_populates="chats")


class Meeting(CustomBase):
    __tablename__ = 'meeting'

    meeting_id = CustomColumn(UUID, primary_key=True, default=lambda: uuid.uuid4(), label="The chat's unique identifier (UUID).")
    meeting_name = CustomColumn(String, nullable=False, label="The name of the meeting.")
    meeting_timestamp = CustomColumn(DateTime, default=datetime.datetime.now(), label="The timestamp of the meeting.")

    # Relationships
    chats = relationship("Chat", back_populates="meeting")
    agents = relationship("Agent", 
                        secondary="agents_by_meeting",
                        back_populates="meetings")


class AgentsbyMeeting(CustomBase):
    __tablename__ = 'agents_by_meeting'

    agent_id = CustomColumn(String, ForeignKey('agent.agent_id'), primary_key=True, label="The agent's unique identifier (UUID).")
    meeting_id = CustomColumn(UUID, ForeignKey('meeting.meeting_id'), primary_key=True, label="The chat's unique identifier (UUID).")
    agents_by_meeting_timestamp = CustomColumn(DateTime, default=datetime.datetime.now(), label="The timestamp of the agent's addition to the meeting.")

class Agent(CustomBase):
    __tablename__ = 'agent'

    agent_id = CustomColumn(UUID, primary_key=True, default=lambda: uuid.uuid4(), label="The agent's unique identifier (UUID).")
    agent_name = CustomColumn(String, nullable=False, label="The agent's name.")
    role = CustomColumn(String, nullable=False, label="The agent's role.")
    model = CustomColumn(String, nullable=False, label="The LLM model to be used.")
    temperature = CustomColumn(Float, nullable=False, default=0.7, label="The sampling temperature. The higher the temperature, the more creative the responses.")
    agent_timestamp = CustomColumn(DateTime, default=datetime.datetime.now(), label="The timestamp of the agent's creation.")

    # Relationships
    chats = relationship("Chat", back_populates="agent")
    meetings = relationship("Meeting",
                          secondary="agents_by_meeting",
                          back_populates="agents")

    def __init__(self, agent_name, role='helpful assistant', model='gpt-4o-mini', temperature=0.5):
        super().__init__(agent_name=agent_name, role=role, model=model, temperature=temperature)
        characters = string.ascii_letters + string.digits  # includes both upper/lower case letters and numbers
        random_id = ''.join(random.choices(characters, k=4))
        self.agent_name = agent_name + " " + random_id

    def __repr__(self):
        return f"{self.agent_name} {self.agent_id}"
    
    
    @property
    def chat_history(self):
        meetings = self.meetings
        chats = []
        for meeting in meetings:
            chats.extend(meeting.chats)

        # order chats by timestamp
        chats = sorted(chats, key=lambda x: x.chat_timestamp)

        # Convert into the format [{role: agent, content: chat_content}]
        def to_chat(chat):
            if chat.agent.agent_id == self.agent_id:
                role = "assistant"
                content = "You: " + chat.content
            elif chat.agent.agent_name == "system":
                role = "system"
                content = "System: " + chat.content

            else:
                role = "user"
                content = chat.agent.agent_name + ": " + chat.content

            return {"role": role, "content": content}


        chats = [to_chat(chat) for chat in chats]
        return chats
        

    def forward(self, response_format) -> dict:

        messages = self.chat_history

        response_json = get_structured_json_response_from_gpt(
            messages=messages,
            response_format=response_format,
            model='gpt-4o-mini',
            temperature=0.5
        )

        ic(response_json)

        return response_json



if __name__ == '__main__':

    # Create objects
    agent1 = Agent(agent_name="Agent 1")
    agent2 = Agent(agent_name="Agent 2")
    meeting = Meeting(meeting_name="First Meeting")
    chat1 = Chat(content="Hello", agent=agent1, meeting=meeting)
    print(chat1.to_dict())
    chat2 = Chat(content="Hi", agent=agent2, meeting=meeting)

    # Add agent to meeting
    meeting.agents.append(agent1)
    meeting.agents.append(agent2)

    # Get all chats for an agent
    print(agent1.chat_history)

    # # Get all meetings for an agent
    # print(agent1.meetings)

    # # Get all agents in a meeting
    # print(meeting.agents)
