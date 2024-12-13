import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .base import Base, Wrapper  # noqa

from .tables import (
    Framework,
    Cluster,
    Generation,
    Population,
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
