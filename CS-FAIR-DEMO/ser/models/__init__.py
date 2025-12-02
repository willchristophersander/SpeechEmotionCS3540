"""Model architectures for speech emotion recognition."""

from .crnn import CRNN
from .crnn_6class import CRNN_6Class, EMOTIONS_6CLASS, NUM_CLASSES_6CLASS, create_6class_model

__all__ = ['CRNN', 'CRNN_6Class', 'EMOTIONS_6CLASS', 'NUM_CLASSES_6CLASS', 'create_6class_model']

