"""Loss functions for speech emotion recognition."""

from .distance_weighted import DistanceWeightedLoss, EMOTION_DISTANCE_MATRIX
from .distance_weighted_6class import DistanceWeightedLoss as DistanceWeightedLoss6Class, EMOTION_DISTANCE_MATRIX_6CLASS

__all__ = [
    'DistanceWeightedLoss', 'EMOTION_DISTANCE_MATRIX',
    'DistanceWeightedLoss6Class', 'EMOTION_DISTANCE_MATRIX_6CLASS'
]

