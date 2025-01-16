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
from .system import Agent, Chat, Meeting


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
    system_capability_ci_lower = CustomColumn(Float, label="")
    system_capability_ci_upper = CustomColumn(Float, label="")
    system_capability_ci_median = CustomColumn(Float, label="")
    system_capability_ci_sample_size = CustomColumn(Float, label="")
    system_capability_ci_confidence_level = CustomColumn(Float, label="")
    system_safety_ci_lower = CustomColumn(Float, label="")
    system_safety_ci_upper = CustomColumn(Float, label="")
    system_safety_ci_median = CustomColumn(Float, label="")
    system_safety_ci_sample_size = CustomColumn(Float, label="")
    system_safety_ci_confidence_level = CustomColumn(Float, label="")
    cluster_id = CustomColumn(
        String,
        ForeignKey("cluster.cluster_id"),
        label="The cluster's unique identifier (UUID).",
    )
    generation_timestamp = CustomColumn(
        DateTime,
        label="The generation's timestamp.",
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
            .order_by(System.system_capability_ci_median.desc())
            .first()
        )

        if not elite:
            raise ValueError("No elite found in cluster.")

        # print("Elite: ", elite)

        return elite

    @property
    def pareto_elites(self):
        """
        Return all Pareto-optimal systems in this cluster with respect to
        (system_fitness, system_safety).
        """
        session = object_session(self)
        # Fetch all systems in this cluster
        systems_in_cluster = (
            session.query(System).filter(System.cluster_id == self.cluster_id).all()
        )

        if not systems_in_cluster:
            raise ValueError("No systems found in this cluster.")

        def dominates(s1, s2):
            """
            Returns True if s1 dominates s2 across the two objectives:
            - system_capability_ci_median
            - system_safety_ci_median
            We assume we are maximizing both objectives.
            """
            if not s1.system_capability_ci_median:
                return False
            elif not s2.system_capability_ci_median:
                return True

            return (
                s1.system_capability_ci_median >= s2.system_capability_ci_median
                and s1.system_safety_ci_median >= s2.system_safety_ci_median
                and (
                    s1.system_capability_ci_median > s2.system_capability_ci_median
                    or s1.system_safety_ci_median > s2.system_safety_ci_median
                )
            )

        # Compute the Pareto front (non-dominated set)
        pareto_front = []
        for s1 in systems_in_cluster:
            # Check if s1 is dominated by any other system
            is_dominated = False
            for s2 in systems_in_cluster:
                if s1 == s2:
                    continue
                if dominates(s2, s1):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_front.append(s1)

        print(
            "Pareto front: ",
            [
                {
                    "system_id": p.system_id,
                    "capability": p.system_capability_ci_median,
                    "safety": p.system_safety_ci_median,
                }
                for p in pareto_front
            ],
        )

        return pareto_front


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
    def pareto_elites(self) -> list[System]:
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

        # print("Generation", most_recent_generation.generation_id)

        assert len(most_recent_generation.clusters) > 0

        # Find the elites from each cluster
        elites = []
        for cluster in most_recent_generation.clusters:
            elites.extend(cluster.pareto_elites)

        return elites

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

        # print("Generation", most_recent_generation.generation_id)

        assert len(most_recent_generation.clusters) > 0

        # Find the elites from each cluster
        elites = [cluster.elite for cluster in most_recent_generation.clusters]

        assert len(elites) == len(most_recent_generation.clusters)

        # print("Elites: ", elites)

        return elites
