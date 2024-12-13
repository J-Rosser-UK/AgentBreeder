from sqlalchemy import Column, String, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
import datetime
import uuid
import random
import string
from sqlalchemy.orm import object_session
from .base import CustomBase, CustomColumn, AutoSaveList
from chat import get_structured_json_response_from_gpt


class Framework(CustomBase):
    __tablename__ = "framework"

    framework_id = CustomColumn(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        label="The framework's unique identifier (UUID).",
    )
    framework_timestamp = CustomColumn(
        DateTime,
        default=datetime.datetime.utcnow,
        label="The timestamp of the framework.",
    )
    framework_name = CustomColumn(String, label="The name of the framework.")
    framework_code = CustomColumn(
        String,
        label="The code of the framework. Starting with def forward(self, task: str) -> str:",
    )
    framework_thought_process = CustomColumn(
        String, label="The thought process that went into creating the framework."
    )
    population_id = CustomColumn(
        String,
        ForeignKey("population.population_id"),
        label="The population's unique identifier (UUID).",
    )
    framework_fitness = CustomColumn(Float, label="The fitness of the framework.")
    framework_descriptor = CustomColumn(
        JSON, label="The embedding of the framework as a list of floats."
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

    # Relationships
    meetings = relationship(
        "Meeting", back_populates="framework", collection_class=AutoSaveList
    )
    population = relationship(
        "Population", back_populates="frameworks", collection_class=AutoSaveList
    )
    cluster = relationship(
        "Cluster", back_populates="frameworks", collection_class=AutoSaveList
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
    frameworks = relationship(
        "Framework", back_populates="cluster", collection_class=AutoSaveList
    )

    @property
    def elite(self):
        """
        Returns the framework with the highest framework_fitness in the cluster.
        If no frameworks are associated with the cluster, returns None.
        """
        # from sqlalchemy.orm.session import object_session  # Ensure we use the correct function

        # Get the session associated with this object
        session = object_session(self)

        # Query the Framework table for the highest fitness framework in this cluster
        elite = (
            session.query(Framework)
            .filter(Framework.cluster_id == self.cluster_id)
            .order_by(Framework.framework_fitness.desc())
            .first()
        )

        print("Elite: ", elite)

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

    # Relationships
    frameworks = relationship(
        "Framework", back_populates="population", collection_class=AutoSaveList
    )
    clusters = relationship(
        "Cluster", back_populates="population", collection_class=AutoSaveList
    )
    generations = relationship(
        "Generation", back_populates="population", collection_class=AutoSaveList
    )

    @property
    def elites(self) -> list[Framework]:
        """Returns from the most recent generation the elites from each cluster."""

        session = object_session(self)

        # Find most recent generation
        most_recent_generation = (
            session.query(Generation)
            .filter_by(population_id=self.population_id)
            .order_by(Generation.generation_id.desc())
            .first()
        )

        if not most_recent_generation:
            elites = self.frameworks
            assert len(elites) > 0
            return elites

        print("Generation", most_recent_generation.generation_id)

        assert len(most_recent_generation.clusters) > 0

        # Find the elites from each cluster
        elites = [cluster.elite for cluster in most_recent_generation.clusters]

        assert len(elites) == len(most_recent_generation.clusters)

        print("Elites: ", elites)

        return elites
