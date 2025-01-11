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

load_dotenv(override=True)

# Create engine and Base
current_dir = os.path.dirname(os.path.abspath(__file__))
engine = create_engine(
    os.getenv("DATABASE_URL"),
    pool_size=100,  # Number of connections in the pool
    max_overflow=10,  # Additional connections allowed beyond pool_size
)


def initialize_session():
    """
    Returns a new thread-safe session.
    """

    # Session factory
    SessionFactory = sessionmaker(bind=engine)

    # Create tables
    Base.metadata.create_all(engine)

    assert len(Base.metadata.tables.keys()) > 0

    session = SessionFactory()

    try:
        yield session
    except:
        session.rollback()
    finally:
        session.commit()
        session.close()
