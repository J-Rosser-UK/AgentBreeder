import uuid
import datetime
import random
import string
from sqlalchemy import inspect
import inspect as py_inspect  # To avoid naming conflict with sqlalchemy.inspect
from typing import Callable
from base import Agent, Meeting, Chat  # Ensure these are properly imported

def initialize_attributes_from_model(self, model_class, special_defaults={}):
    mapper = inspect(model_class)
    for column in mapper.columns:
        attr_name = column.key
        default = column.default

        default_value = None  # Initialize default_value

        # Handle special defaults
        if attr_name in special_defaults:
            default_value = special_defaults[attr_name]
        elif default is not None:
            if default.is_scalar:
                default_value = default.arg
            elif default.is_callable:
                # Attempt to call default.arg if possible
                if isinstance(default.arg, Callable):
                    sig = py_inspect.signature(default.arg)
                    if len(sig.parameters) == 0:
                        default_value = default.arg()
                    else:
                        # Can't call callable with parameters
                        default_value = None
                else:
                    default_value = None
            else:
                default_value = None
        else:
            default_value = None

        setattr(self, attr_name, default_value)

class MockAgent:
    def __init__(self, agent_name, model='gpt-4o-mini', temperature=0.5):
        # Initialize attributes from Agent's columns
        special_defaults = {
            'agent_id': str(uuid.uuid4()),
            'agent_timestamp': datetime.datetime.now(),
        }
        initialize_attributes_from_model(self, Agent, special_defaults)

        # Set provided parameters
        self.agent_name = agent_name
        self.model = model
        self.temperature = temperature

        # Generate random ID and append to agent_name
        characters = string.ascii_letters + string.digits
        random_id = ''.join(random.choices(characters, k=4))
        self.agent_name = f"{agent_name} {random_id}"

        # Initialize relationships
        self.chats = []
        self.meetings = []

    def __repr__(self):
        return f"{self.agent_name} {self.agent_id}"

    @property
    def chat_history(self):
        # Mocked chat history
        return []

    def forward(self, response_format) -> dict:
        # Generate a random JSON response matching the response_format keys
        response = {}
        for key in response_format:
            response[key] = f"Random {key} value"
        return response

class MockMeeting:
    def __init__(self, meeting_name):
        # Initialize attributes from Meeting's columns
        special_defaults = {
            'meeting_id': str(uuid.uuid4()),
            'meeting_timestamp': datetime.datetime.now(),
        }
        initialize_attributes_from_model(self, Meeting, special_defaults)

        # Set provided parameters
        self.meeting_name = meeting_name

        # Initialize relationships
        self.framework = None
        self.chats = []
        self.agents = []

    def __repr__(self):
        return f"Meeting {self.meeting_name} {self.meeting_id}"

class MockChat:
    def __init__(self, content, agent=None, meeting=None):
        # Initialize attributes from Chat's columns
        special_defaults = {
            'chat_id': str(uuid.uuid4()),
            'chat_timestamp': datetime.datetime.now(),
        }
        initialize_attributes_from_model(self, Chat, special_defaults)

        # Set provided parameters
        self.content = content
        if agent is not None:
            self.agent = agent
            self.agent_id = agent.agent_id
        else:
            self.agent = None
            self.agent_id = None

        if meeting is not None:
            self.meeting = meeting
            self.meeting_id = meeting.meeting_id
        else:
            self.meeting = None
            self.meeting_id = None

    def __repr__(self):
        return f"Chat {self.chat_id}: {self.content}"

# Example usage:

if __name__ == "__main__":
    # Create some agents
    agent1 = MockAgent(agent_name="Agent 1")
    agent2 = MockAgent(agent_name="Agent 2")

    # Create a meeting
    meeting = MockMeeting(meeting_name="Strategy Meeting")

    # Add agents to the meeting
    meeting.agents.append(agent1)
    meeting.agents.append(agent2)

    # Create some chats
    chat1 = MockChat(content="Hello, how are you?", agent=agent1, meeting=meeting)
    chat2 = MockChat(content="I'm good, thank you!", agent=agent2, meeting=meeting)

    # Add chats to meeting
    meeting.chats.append(chat1)
    meeting.chats.append(chat2)

    # Add chats to agents
    agent1.chats.append(chat1)
    agent2.chats.append(chat2)

    # Print meeting details
    print(meeting)
    print(f"Agents in meeting: {[agent.agent_name for agent in meeting.agents]}")
    print(f"Chats in meeting: {[chat.content for chat in meeting.chats]}")

    # Print agent details
    print(agent1)
    print(f"Chats of {agent1.agent_name}: {[chat.content for chat in agent1.chats]}")

    # Print chat details
    print(chat1)
    print(f"Chat agent: {chat1.agent.agent_name}")
    print(f"Chat meeting: {chat1.meeting.meeting_name}")
