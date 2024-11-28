from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from concurrent.futures import ThreadPoolExecutor
import threading
from sqlalchemy.orm import declarative_base

# Define the database model
Base = declarative_base()

class ExampleModel(Base):
    __tablename__ = 'example'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    counter = Column(Integer, default=0)  # Counter to demonstrate updates

# Set up the SQLite engine with check_same_thread=False
engine = create_engine('sqlite:///example.db', connect_args={'check_same_thread': False})

# Create tables
Base.metadata.create_all(engine)

# Create a sessionmaker factory
SessionFactory = sessionmaker(bind=engine)

# Initialize the ExampleModel entry if not already present
def initialize_database():
    session = SessionFactory()
    try:
        if not session.query(ExampleModel).first():
            example_entry = ExampleModel(name="Initial Name", counter=0)
            session.add(example_entry)
            session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error during initialization: {e}")
    finally:
        session.close()

def thread_safe_session():
    """
    This function creates a new session for the current thread and updates the shared model.
    """
    session = SessionFactory()
    try:
        # Fetch the first ExampleModel entry
        example_entry = session.query(ExampleModel).first()
        if example_entry:
            # Update the counter and name
            example_entry.counter += 1
            example_entry.name = f"Updated by {threading.current_thread().name}"
            session.commit()
            print(f"{threading.current_thread().name}: Counter updated to {example_entry.counter}")
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()

# Use ThreadPoolExecutor for concurrency
def main():
    initialize_database()
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(thread_safe_session) for _ in range(10)]
        for future in futures:
            future.result()

if __name__ == "__main__":
    main()
