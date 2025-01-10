from sqlalchemy import Column, String, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
import datetime
import uuid
import random
import string
from sqlalchemy.orm import object_session
from .base import CustomBase, CustomColumn, AutoSaveList
from chat import get_structured_json_response_from_gpt
import asyncio
from functools import wraps
import threading


class Chat(CustomBase):
    __tablename__ = "chat"

    chat_id = CustomColumn(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        label="The chat's unique identifier (UUID).",
    )
    agent_id = CustomColumn(
        String, ForeignKey("agent.agent_id"), label="The role of the chat."
    )
    meeting_id = CustomColumn(
        String,
        ForeignKey("meeting.meeting_id"),
        label="The meeting's unique identifier (UUID).",
    )
    content = CustomColumn(String, label="The content of the chat.")
    chat_timestamp = CustomColumn(
        DateTime, default=datetime.datetime.utcnow, label="The timestamp of the chat."
    )

    # Relationships
    agent = relationship("Agent", back_populates="chats", collection_class=AutoSaveList)
    meeting = relationship(
        "Meeting", back_populates="chats", collection_class=AutoSaveList
    )


class System(CustomBase):
    __tablename__ = "system"

    system_id = CustomColumn(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        label="The system's unique identifier (UUID).",
    )

    system_first_parent_id = CustomColumn(
        String,
        label="The first parent's unique identifier (UUID).",
    )

    system_second_parent_id = CustomColumn(
        String,
        label="The second parent's unique identifier (UUID). This may be None if mutation rather than crossover.",
    )

    system_timestamp = CustomColumn(
        DateTime,
        default=datetime.datetime.utcnow,
        label="The timestamp of the multi-agent system.",
    )
    system_name = CustomColumn(String, label="The name of the multi-agent system.")
    system_code = CustomColumn(
        String,
        label="The code of the multi-agent system. Starting with def forward(self, task: str) -> str:",
    )
    system_thought_process = CustomColumn(
        String,
        label="The thought process that went into creating the multi-agent system.",
    )
    population_id = CustomColumn(
        String,
        ForeignKey("population.population_id"),
        label="The population's unique identifier (UUID).",
    )
    system_fitness = CustomColumn(Float, label="The fitness of the multi-agent system.")
    system_descriptor = CustomColumn(
        JSON, label="The embedding of the multi-agent system as a list of floats."
    )
    ci_lower = CustomColumn(Float, label="")
    ci_upper = CustomColumn(Float, label="")
    ci_median = CustomColumn(Float, label="")
    ci_sample_size = CustomColumn(Float, label="")
    ci_confidence_level = CustomColumn(Float, label="")
    cluster_id = CustomColumn(
        String,
        ForeignKey("cluster.cluster_id"),
        label="The cluster's unique identifier (UUID).",
    )
    generation_timestamp = CustomColumn(
        DateTime,
        label="The generation's timestamp.",
    )

    # Relationships
    meetings = relationship(
        "Meeting", back_populates="system", collection_class=AutoSaveList
    )
    population = relationship(
        "Population", back_populates="systems", collection_class=AutoSaveList
    )
    cluster = relationship(
        "Cluster", back_populates="systems", collection_class=AutoSaveList
    )


class Cluster(CustomBase):
    __tablename__ = "cluster"

    cluster_id = CustomColumn(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        label="The cluster's unique identifier (UUID).",
    )
    cluster_timestamp = CustomColumn(
        DateTime,
        default=datetime.datetime.utcnow,
        label="The timestamp of the cluster.",
    )
    cluster_name = CustomColumn(String, label="The name of the cluster.")
    cluster_description = CustomColumn(String, label="The description of the cluster.")
    generation_id = CustomColumn(
        String,
        ForeignKey("generation.generation_id"),
        label="The generation's unique identifier (UUID).",
    )
    population_id = CustomColumn(
        String,
        ForeignKey("population.population_id"),
        label="The population's unique identifier (UUID).",
    )

    # Relationships
    population = relationship(
        "Population", back_populates="clusters", collection_class=AutoSaveList
    )
    generation = relationship(
        "Generation", back_populates="clusters", collection_class=AutoSaveList
    )
    systems = relationship(
        "System", back_populates="cluster", collection_class=AutoSaveList
    )

    @property
    def elite(self):
        """
        Returns the multi-agent system with the highest system_fitness in the cluster.
        If no systems are associated with the cluster, returns None.
        """
        # from sqlalchemy.orm.session import object_session  # Ensure we use the correct function

        # Get the session associated with this object
        session = object_session(self)

        # print("Cluster ID: ", self.cluster_id)
        # print("Session: ", session)

        # Query the System table for the highest fitness system in this cluster
        elite = (
            session.query(System)
            .filter(System.cluster_id == self.cluster_id)
            .order_by(System.system_fitness.desc())
            .first()
        )

        if not elite:
            raise ValueError("No elite found in cluster.")

        # print("Elite: ", elite)

        return elite


class Generation(CustomBase):
    __tablename__ = "generation"

    generation_id = CustomColumn(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        label="The generation's unique identifier (UUID).",
    )
    generation_timestamp = CustomColumn(
        DateTime,
        default=datetime.datetime.utcnow,
        label="The timestamp of the generation.",
    )
    population_id = CustomColumn(
        String,
        ForeignKey("population.population_id"),
        label="The population's unique identifier (UUID).",
    )

    # Relationships
    population = relationship(
        "Population", back_populates="generations", collection_class=AutoSaveList
    )
    clusters = relationship(
        "Cluster", back_populates="generation", collection_class=AutoSaveList
    )


class Population(CustomBase):
    __tablename__ = "population"

    population_id = CustomColumn(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        label="The population's unique identifier (UUID).",
    )
    population_timestamp = CustomColumn(
        DateTime,
        default=datetime.datetime.utcnow,
        label="The timestamp of the population.",
    )

    population_benchmark = CustomColumn(
        String,
        label="The benchmark name.",
    )

    # Relationships
    systems = relationship(
        "System", back_populates="population", collection_class=AutoSaveList
    )
    clusters = relationship(
        "Cluster", back_populates="population", collection_class=AutoSaveList
    )
    generations = relationship(
        "Generation", back_populates="population", collection_class=AutoSaveList
    )

    @property
    def elites(self) -> list[System]:
        """Returns from the most recent generation the elites from each cluster."""

        session = object_session(self)

        # Find most recent generation
        most_recent_generation = (
            session.query(Generation)
            .filter_by(population_id=self.population_id)
            .order_by(Generation.generation_timestamp.desc())
            .first()
        )

        if not most_recent_generation:
            elites = self.systems
            assert len(elites) > 0
            return elites

        print("Generation", most_recent_generation.generation_id)

        assert len(most_recent_generation.clusters) > 0

        # Find the elites from each cluster
        elites = [cluster.elite for cluster in most_recent_generation.clusters]

        assert len(elites) == len(most_recent_generation.clusters)

        # print("Elites: ", elites)

        return elites


class Meeting(CustomBase):
    __tablename__ = "meeting"

    meeting_id = CustomColumn(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        label="The chat's unique identifier (UUID).",
    )
    meeting_name = CustomColumn(String, label="The name of the meeting.")
    meeting_timestamp = CustomColumn(
        DateTime,
        default=datetime.datetime.utcnow,
        label="The timestamp of the meeting.",
    )
    system_id = CustomColumn(
        String,
        ForeignKey("system.system_id"),
        label="The system's unique identifier (UUID).",
    )

    # Relationships
    system = relationship("System", back_populates="meetings")
    chats = relationship(
        "Chat", back_populates="meeting", collection_class=AutoSaveList
    )
    agents = relationship(
        "Agent",
        secondary="agents_by_meeting",
        back_populates="meetings",
        collection_class=AutoSaveList,
    )


class AgentsbyMeeting(CustomBase):
    __tablename__ = "agents_by_meeting"

    agent_id = CustomColumn(
        String,
        ForeignKey("agent.agent_id"),
        primary_key=True,
        label="The agent's unique identifier (UUID).",
    )
    meeting_id = CustomColumn(
        String,
        ForeignKey("meeting.meeting_id"),
        primary_key=True,
        label="The chat's unique identifier (UUID).",
    )
    agents_by_meeting_timestamp = CustomColumn(
        DateTime,
        default=datetime.datetime.utcnow,
        label="The timestamp of the agent's addition to the meeting.",
    )


class Agent(CustomBase):
    __tablename__ = "agent"

    agent_id = CustomColumn(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        label="The agent's unique identifier (UUID).",
    )
    agent_name = CustomColumn(String, label="The agent's name.")
    agent_backstory = CustomColumn(
        String, label="A long description of the agent's backstory."
    )
    model = CustomColumn(String, label="The LLM model to be used.")
    temperature = CustomColumn(
        Float,
        default=0.7,
        label="The sampling temperature. The higher the temperature, the more creative the responses.",
    )
    agent_timestamp = CustomColumn(
        DateTime,
        default=datetime.datetime.utcnow,
        label="The timestamp of the agent's creation.",
    )

    # Relationships
    chats = relationship("Chat", back_populates="agent", collection_class=AutoSaveList)
    meetings = relationship(
        "Meeting",
        secondary="agents_by_meeting",
        back_populates="agents",
        collection_class=AutoSaveList,
    )

    def __init__(self, session, agent_name, model="gpt-4o-mini", temperature=0.5):
        super().__init__(
            session, agent_name=agent_name, model=model, temperature=temperature
        )
        characters = (
            string.ascii_letters + string.digits
        )  # includes both upper/lower case letters and numbers
        random_id = "".join(random.choices(characters, k=4))
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

    async def forward(self, response_format) -> dict:

        # logging.info(f"Agent {self.agent_name} is thinking...")

        messages = self.chat_history

        response_json = await get_structured_json_response_from_gpt(
            messages=messages, response_format=response_format, temperature=0.5
        )

        # logging.info(f"Agent {self.agent_name} has responded with: \n{response_json}\n -------------------")

        return response_json
