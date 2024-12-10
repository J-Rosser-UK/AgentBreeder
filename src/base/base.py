from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy import (
    Column,
    ForeignKey,
    DateTime,
    Integer,
    Boolean,
    Float,
    JSON,
    String,
)
from sqlalchemy.orm import object_session
import datetime
from sqlalchemy.orm import Session
from sqlalchemy.sql.sqltypes import String, DateTime

Base = declarative_base()


class Wrapper:
    """
    A class that wraps another class, injecting a session into its constructor.

    Attributes:
        cls (type): The class to be wrapped.
        session (Session): The session to be injected into the wrapped class.

    Methods:
        __call__(*args, **kwargs): Calls the wrapped class with the provided arguments,
                                   injecting the session into the keyword arguments.
    """

    def __init__(self, cls, session: Session):
        self.cls = cls
        self.session = session

    def __call__(self, *args, **kwargs):
        kwargs["session"] = self.session
        return self.cls(*args, **kwargs)


class CustomBase(Base):
    __abstract__ = True

    def __init__(self, session=None, **kwargs):
        # Initialize the extra attributes dictionary first
        self._extra_attrs = {}

        # Validate and set default values for columns
        for column in self.__table__.columns:
            column_name = column.name
            column_type = column.type

            # Set default values if defined
            if column.default is not None:
                if column.default.is_scalar:
                    setattr(self, column_name, column.default.arg)
                elif isinstance(column_type, DateTime):
                    setattr(self, column_name, datetime.datetime.utcnow())

            # Validate provided values
            if column_name in kwargs:
                self.validate_column_value(
                    column_name, column_type, kwargs[column_name]
                )

        # Set provided values after validation
        for attr, value in kwargs.items():
            setattr(self, attr, value)

        if session:
            self.session = session
            # Add self to the session and commit

            session.add(self)

            session.commit()

    def validate_column_value(self, column_name, column_type, value):
        """
        Validates the value of a column against its expected type.
        """
        if isinstance(column_type, String):
            if not isinstance(value, str):
                raise Exception(
                    f"Invalid value for column '{column_name}'. Expected a string but got {type(value).__name__}."
                )
        elif isinstance(column_type, Float):
            if not isinstance(
                value, (float, int)
            ):  # Allow integers since they can be cast to float
                raise Exception(
                    f"Invalid value for column '{column_name}'. Expected a float but got {type(value).__name__}."
                )
        elif isinstance(column_type, JSON):
            if not isinstance(value, (dict, list, str, int, float, bool, type(None))):
                raise Exception(
                    f"Invalid value for column '{column_name}'. Expected a JSON-compatible type but got {type(value).__name__}."
                )
        elif isinstance(column_type, ForeignKey):
            if not isinstance(
                value, str
            ):  # Typically foreign keys are strings (UUID or other identifiers)
                raise Exception(
                    f"Invalid value for foreign key column '{column_name}'. Expected a string but got {type(value).__name__}."
                )
        elif isinstance(column_type, DateTime):
            if not isinstance(
                value, (datetime.datetime, str)
            ):  # Allow datetime or ISO 8601 string
                raise Exception(
                    f"Invalid value for column '{column_name}'. Expected a datetime or ISO 8601 string but got {type(value).__name__}."
                )
        elif isinstance(column_type, Integer):
            if not isinstance(value, int):
                raise Exception(
                    f"Invalid value for column '{column_name}'. Expected an integer but got {type(value).__name__}."
                )
        elif isinstance(column_type, Boolean):
            if not isinstance(value, bool):
                raise Exception(
                    f"Invalid value for column '{column_name}'. Expected a boolean but got {type(value).__name__}."
                )
        else:
            raise Exception(
                f"Unsupported column type '{type(column_type).__name__}' for column '{column_name}'."
            )

    def to_dict(self):
        # Include extra attributes in the dictionary representation
        obj_dict = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        obj_dict.update(getattr(self, "_extra_attrs", {}))
        return obj_dict

    def update(self, **kwargs):
        session = object_session(self)
        for attr, value in kwargs.items():
            setattr(self, attr, value)
        session.add(self)
        session.commit()


class AutoSaveList(list):
    def append(self, item):
        super().append(item)
        if item:
            session = object_session(item)
            if session:
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
