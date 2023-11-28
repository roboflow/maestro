from enum import Enum


class Mode(Enum):
    """
    Enumeration for different marking modes.
    """
    NUMERIC = "NUMERIC"
    ALPHABETIC = "ALPHABETIC"

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))
