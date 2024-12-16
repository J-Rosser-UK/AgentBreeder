import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from .base import Base, Wrapper  # noqa

from .tables import (
    Chat,
    Framework,
    Cluster,
    Generation,
    Population,
    Meeting,
    AgentsbyMeeting,
    Agent,
)  # noqa


def initialize_session(db_name: str):
    """
    Returns a new thread-safe session.
    """

    # Create engine and Base
    current_dir = os.path.dirname(os.path.abspath(__file__))
    engine = create_engine(
        f"sqlite:///{current_dir}/db/{db_name}",
        connect_args={"check_same_thread": False},
    )

    # Session factory
    SessionFactory = sessionmaker(bind=engine)

    # Create tables
    Base.metadata.create_all(engine)
    # print(Base.metadata.tables.keys())

    assert len(Base.metadata.tables.keys()) > 0

    return SessionFactory(), Base


def initialize_async_session(db_name: str):
    # Initialize async engine and session
    current_dir = os.path.dirname(os.path.abspath(__file__))
    async_engine = create_async_engine(
        f"sqlite+aiosqlite:///{current_dir}/db/{db_name}",
        echo=True,  # Optional: Enable SQL query logging
        future=True,
    )
    AsyncSessionLocal = sessionmaker(
        bind=async_engine, class_=AsyncSession, expire_on_commit=False
    )
    return AsyncSessionLocal()
