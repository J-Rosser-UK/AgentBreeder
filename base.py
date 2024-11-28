from sqlalchemy import Column, String, Float, JSON, ForeignKey, DateTime, Integer, Boolean
import datetime
from sqlalchemy.orm import declarative_base, relationship
import uuid
from chat import get_structured_json_response_from_gpt
from icecream import ic
import random
import string
import os
from sqlalchemy.orm.collections import collection

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import logging
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.orm.session import object_session
from gemini import get_structured_json_response_from_gemini

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
import os


Base = declarative_base()
# Create engine and Base
current_dir = os.path.dirname(os.path.abspath(__file__))
engine = create_engine(f'sqlite:///{current_dir}/chat_database.db', connect_args={"check_same_thread": False})




# Session factory
SessionFactory = sessionmaker(bind=engine)

class CustomBase(Base):
    __abstract__ = True

    def __init__(self, session, **kwargs):
        # Initialize the extra attributes dictionary first
        self._extra_attrs = {}

        # Set default values for columns
        for column in self.__table__.columns:
            if column.default is not None:
                if column.default.is_scalar:
                    setattr(self, column.name, column.default.arg)
                elif isinstance(column.type, DateTime):
                    setattr(self, column.name, datetime.datetime.utcnow())

        # Override with any provided values
        for attr, value in kwargs.items():
            setattr(self, attr, value)

        # Add self to the session and commit
        session.add(self)
        session.commit()

    # def __getattr__(self, name):
    #     # Check if the attribute exists in _extra_attrs
    #     if '_extra_attrs' in self.__dict__ and name in self._extra_attrs:
    #         return self._extra_attrs[name]
    #     raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    # def __setattr__(self, name, value):
    #     # Handle dynamically added attributes
    #     if name not in self.__table__.columns and not name.startswith("_"):
    #         if '_extra_attrs' not in self.__dict__:
    #             self._extra_attrs = {}
    #         self._extra_attrs[name] = value
    #     else:
    #         super().__setattr__(name, value)

    def to_dict(self):
        # Include extra attributes in the dictionary representation
        obj_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        obj_dict.update(getattr(self, "_extra_attrs", {}))
        return obj_dict

    def update(self, session, **kwargs):
        for attr, value in kwargs.items():
            setattr(self, attr, value)
        session.add(self)
        session.commit()
    



class AutoSaveList(list):
    def append(self, item):
        super().append(item)
        session = object_session(item)
        session.add(item)
        session.commit()

    def extend(self, items):
        super().extend(items)
        session = object_session(items[0])
        session.add_all(items)
        session.commit()

class CustomColumn(Column):
    def __init__(self, *args, label=None, **kwargs):
        self.label = label
        super().__init__(*args, **kwargs)
    

class Chat(CustomBase):
    __tablename__ = 'chat'

    chat_id = CustomColumn(String, primary_key=True, default=lambda: str(uuid.uuid4()), label="The chat's unique identifier (UUID).")
    agent_id = CustomColumn(String, ForeignKey('agent.agent_id'), label="The role of the chat.")
    meeting_id = CustomColumn(String, ForeignKey('meeting.meeting_id'), label="The meeting's unique identifier (UUID).")
    content = CustomColumn(String, label="The content of the chat.")
    chat_timestamp = CustomColumn(DateTime, default=datetime.datetime.now(), label="The timestamp of the chat.")

    # Relationships
    agent = relationship("Agent", back_populates="chats", collection_class=AutoSaveList)
    meeting = relationship("Meeting", back_populates="chats",collection_class=AutoSaveList)

class Cluster(CustomBase):
    __tablename__ = 'cluster'

    cluster_id = CustomColumn(String, primary_key=True, default=lambda: str(uuid.uuid4()), label="The cluster's unique identifier (UUID).")
    cluster_timestamp = CustomColumn(DateTime, default=datetime.datetime.now(), label="The timestamp of the cluster.")
    cluster_name = CustomColumn(String, label="The name of the cluster.")
    cluster_description = CustomColumn(String, label="The description of the cluster.")


    @property
    def elite(self):
        """
        Returns the framework with the highest framework_fitness in the cluster.
        If no frameworks are associated with the cluster, returns None.
        """
        # from sqlalchemy.orm.session import object_session  # Ensure we use the correct function

        # Get the session associated with this object
        session = object_session(self)
        if not session:
            raise RuntimeError("Session is not available. Ensure the cluster object is attached to a session.")

        # Query the Framework table for the highest fitness framework in this cluster
        return (
            session.query(Framework)
            .filter(Framework.cluster_id == self.cluster_id)
            .order_by(Framework.framework_fitness.desc())
            .first()
        )



    

class Framework(CustomBase):
    __tablename__ = 'framework'

    framework_id = CustomColumn(String, primary_key=True, default=lambda: str(uuid.uuid4()), label="The framework's unique identifier (UUID).")
    framework_timestamp = CustomColumn(DateTime, default=datetime.datetime.now(), label="The timestamp of the framework.")
    framework_name = CustomColumn(String, label="The name of the framework.")
    framework_code = CustomColumn(String, label="The code of the framework. Starting with def forward(self, task: str) -> str:")
    framework_thought_process = CustomColumn(String, label="The thought process that went into creating the framework.")
    framework_generation = CustomColumn(Integer, label="The generation of the framework.")
    population_id = CustomColumn(String, ForeignKey('population.population_id'), label="The population's unique identifier (UUID).")
    framework_fitness = CustomColumn(Float, label="The fitness of the framework.")
    framework_descriptor = CustomColumn(JSON, label="The embedding of the framework as a list of floats.")
    ci_lower = CustomColumn(Float, label="")
    ci_upper = CustomColumn(Float, label="")
    ci_median = CustomColumn(Float, label="")
    ci_sample_size = CustomColumn(Float, label="")
    ci_confidence_level = CustomColumn(Float, label="")
    cluster_id = CustomColumn(String, ForeignKey('cluster.cluster_id'), label="The cluster's unique identifier (UUID).")
    

    # Relationships
    meetings = relationship("Meeting", back_populates="framework", collection_class=AutoSaveList)
    population = relationship("Population", back_populates="frameworks")


class Population(CustomBase):
    __tablename__ = 'population'    

    population_id = CustomColumn(String, primary_key=True, default=lambda: str(uuid.uuid4()), label="The population's unique identifier (UUID).")
    population_timestamp = CustomColumn(DateTime, default=datetime.datetime.now(), label="The timestamp of the population.")

    # Relationships
    frameworks = relationship("Framework", back_populates="population", collection_class=AutoSaveList)




class Meeting(CustomBase):
    __tablename__ = 'meeting'

    meeting_id = CustomColumn(String, primary_key=True, default=lambda: str(uuid.uuid4()), label="The chat's unique identifier (UUID).")
    meeting_name = CustomColumn(String, label="The name of the meeting.")
    meeting_timestamp = CustomColumn(DateTime, default=datetime.datetime.now(), label="The timestamp of the meeting.")
    framework_id = CustomColumn(String, ForeignKey('framework.framework_id'), label="The framework's unique identifier (UUID).")

    # Relationships
    framework = relationship("Framework", back_populates="meetings")
    chats = relationship("Chat", back_populates="meeting", collection_class=AutoSaveList)
    agents = relationship("Agent", 
                           secondary="agents_by_meeting",
                           back_populates="meetings", 
                           collection_class=AutoSaveList)



class AgentsbyMeeting(CustomBase):
    __tablename__ = 'agents_by_meeting'

    agent_id = CustomColumn(String, ForeignKey('agent.agent_id'), primary_key=True, label="The agent's unique identifier (UUID).")
    meeting_id = CustomColumn(String, ForeignKey('meeting.meeting_id'), primary_key=True, label="The chat's unique identifier (UUID).")
    agents_by_meeting_timestamp = CustomColumn(DateTime, default=datetime.datetime.now(), label="The timestamp of the agent's addition to the meeting.")

class Agent(CustomBase):
    __tablename__ = 'agent'

    agent_id = CustomColumn(String, primary_key=True, default=lambda: str(uuid.uuid4()), label="The agent's unique identifier (UUID).")
    agent_name = CustomColumn(String, label="The agent's name.")
    agent_backstory = CustomColumn(String, label="A long description of the agent's backstory.")
    model = CustomColumn(String, label="The LLM model to be used.")
    temperature = CustomColumn(Float, default=0.7, label="The sampling temperature. The higher the temperature, the more creative the responses.")
    agent_timestamp = CustomColumn(DateTime, default=datetime.datetime.now(), label="The timestamp of the agent's creation.")

    # Relationships
    chats = relationship("Chat", back_populates="agent", collection_class=AutoSaveList)
    meetings = relationship("Meeting",
                          secondary="agents_by_meeting",
                          back_populates="agents", collection_class=AutoSaveList)

    def __init__(self, session, agent_name, model='gpt-4o-mini', temperature=0.5):
        super().__init__(session, agent_name=agent_name, model=model, temperature=temperature)
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
            chat_content:str = chat.content if chat.content else ""

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

        logging.info(f"Agent {self.agent_name} is thinking...")

        messages = self.chat_history

        response_json = get_structured_json_response_from_gemini(
            messages=messages,
            response_format=response_format,
            temperature=0.5
        )

        logging.info(f"Agent {self.agent_name} has responded with: \n{response_json}\n -------------------")

        return response_json
    


# Create tables
Base.metadata.create_all(engine)
print(Base.metadata.tables.keys())
# dict_keys([])



def initialize_session():
    """
    Returns a new thread-safe session.
    """
    
    return SessionFactory(), Base

initialize_session()