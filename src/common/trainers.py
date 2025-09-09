
from abc import ABC, abstractmethod

class BaseTrainer(ABC):
    """
    Abstract base class for a trainer.
    """
    def __init__(self, *args, **kwargs):
        # Allow subclasses to have their own constructors
        pass

    @abstractmethod
    def train(self):
        """
        Train the model.
        """
        pass

    @abstractmethod
    def evaluate(self):
        """
        Evaluate the model.
        """
        pass