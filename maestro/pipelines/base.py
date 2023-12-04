from abc import ABC, abstractmethod


class BasePromptCreator(ABC):

    @abstractmethod
    def create(self, *args, **kwargs):
        pass


class BaseResponseProcessor(ABC):

    @abstractmethod
    def process(self, *args, **kwargs):
        pass
