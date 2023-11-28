from enum import Enum


class MarkMode(Enum):
    """
    An enumeration for different marking modes.
    """
    NUMERIC = "NUMERIC"
    ALPHABETIC = "ALPHABETIC"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
