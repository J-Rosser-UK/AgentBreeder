import asyncio
import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv(override=True)
import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

from .base import Base, Wrapper  # noqa

from .tables import (
    Chat,
    System,
    Cluster,
    Generation,
    Population,
    Meeting,
    AgentsbyMeeting,
    Agent,
)  # noqa

from dotenv import load_dotenv


# Create an async engine
async_engine = create_async_engine(
    os.getenv("ASYNC_DATABASE_URL"),  # Replace with your database URL
    pool_size=100,  # Connection pool size
    max_overflow=50,  # Additional connections beyond pool_size
)


def initialize_session():
    """
    Returns a new thread-safe session.
    """

    # Session factory
    # Create a sessionmaker for async sessions
    AsyncSessionFactory = sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Create tables
    Base.metadata.create_all(async_engine)

    assert len(Base.metadata.tables.keys()) > 0

    session = AsyncSessionFactory()

    try:
        yield session
    except:
        session.rollback()
    finally:
        session.commit()
        session.close()
