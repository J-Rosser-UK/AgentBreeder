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

# Create engine and Base
current_dir = os.path.dirname(os.path.abspath(__file__))
engine = create_engine(
    f"sqlite:///{current_dir}/db/experiment.db",
    connect_args={"check_same_thread": False},
    pool_size=100,
    max_overflow=20,
    pool_timeout=60,
)

with engine.connect() as conn:
    conn.execute(text("PRAGMA journal_mode = WAL"))
    conn.execute(text("PRAGMA synchronous = NORMAL"))


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
        session.close()
