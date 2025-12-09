"""
Unified Multi-Dataset Loader
============================

Combines multiple datasets into a single training set.
Validates that all requested emotions are available in all datasets.

Usage:
    from scripts.data.loaders.unified import load_datasets
    
    # 4-class from 4 English datasets
    paths, labels, mapping, used_datasets = load_datasets(
        emotions=['Anger', 'Happy', 'Neutral', 'Sad'],
        datasets=['CREMA-D', 'RAVDESS', 'SAVEE', 'TESS'],
    )
    
    # 6-class from all datasets (will error if any lacks an emotion)
    paths, labels, mapping, used_datasets = load_datasets(
        emotions=['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Surprise'],
        datasets=['CREMA-D', 'RAVDESS', 'SAVEE', 'TESS', 'IEMOCAP', 'nEMO'],
    )
"""

MODULE_INFO = {
    'description': 'Unified loader that combines multiple datasets with emotion validation',
    'inputs': ['List of emotions', 'List of dataset names'],
    'outputs': ['Combined file paths, labels, emotion mapping, datasets used'],
    'status': 'production'
}

from pathlib import Path
from typing import List, Tuple, Dict, Optional

from . import AVAILABLE_LOADERS
from .base import EmotionNotAvailableError


def load_datasets(
    emotions: List[str],
    datasets: Optional[List[str]] = None,
    data_root: Optional[Path] = None,
    skip_unavailable_datasets: bool = False,
    skip_unavailable_emotions: bool = False,
    include_songs: bool = False,
    return_weights: bool = False,
    verbose: bool = True,
) -> Tuple[List[str], List[int], Dict[str, int], List[str], List[float]]:
    """
    Load data from multiple datasets for specified emotions.
    
    Args:
        emotions: List of emotions to load (e.g., ['Anger', 'Happy', 'Neutral', 'Sad'])
        datasets: List of dataset names to use (default: all available)
        data_root: Path to DataSets directory (default: auto-detect)
        skip_unavailable_datasets: If True, skip datasets that don't exist
        skip_unavailable_emotions: If True, skip emotions not in a dataset
        include_songs: If True, automatically include 'RAVDESS Songs' dataset
        verbose: Print loading progress
    
    Returns:
        Tuple of:
            - file_paths: Combined list of audio file paths
            - labels: Combined list of integer labels
            - emotion_to_idx: Dict mapping emotion name to label index
            - datasets_used: List of dataset names that were actually loaded
            - weights: Combined list of sample weights (all 1.0 if return_weights=False)
    
    Raises:
        EmotionNotAvailableError: If a requested emotion isn't available in a dataset
        ValueError: If no valid data is loaded
    """
    if data_root is None:
        # scripts/data/loaders/unified.py -> scripts/data/loaders -> scripts/data -> scripts -> project_root
        data_root = Path(__file__).resolve().parents[3] / 'DataSets'
    
    if datasets is None:
        datasets = list(AVAILABLE_LOADERS.keys())
    
    # If include_songs=True, add 'RAVDESS Songs' to datasets if not already there
    if include_songs and 'RAVDESS Songs' not in datasets:
        datasets = datasets + ['RAVDESS Songs']
    
    if verbose:
        print("=" * 60)
        print(f"Loading data for emotions: {', '.join(emotions)}")
        print(f"From datasets: {', '.join(datasets)}")
        print("=" * 60)
    
    # Validate datasets exist
    for ds_name in datasets:
        if ds_name not in AVAILABLE_LOADERS:
            raise ValueError(f"Unknown dataset: {ds_name}. Available: {list(AVAILABLE_LOADERS.keys())}")
    
    # Create unified emotion mapping
    emotion_to_idx = {e: i for i, e in enumerate(emotions)}
    
    all_paths = []
    all_labels = []
    all_weights = []
    datasets_used = []
    
    for ds_name in datasets:
        loader_class = AVAILABLE_LOADERS[ds_name]
        loader = loader_class(data_root)
        
        try:
            paths, labels, _, weights = loader.load(
                emotions, 
                skip_missing=skip_unavailable_emotions,
                return_weights=return_weights
            )
            
            if paths:
                all_paths.extend(paths)
                all_labels.extend(labels)
                all_weights.extend(weights)
                datasets_used.append(ds_name)
                
                if verbose:
                    # Count per emotion
                    counts = {}
                    for label in labels:
                        emo = emotions[label]
                        counts[emo] = counts.get(emo, 0) + 1
                    counts_str = ', '.join(f"{e}:{c}" for e, c in sorted(counts.items()))
                    print(f"    → {len(paths)} samples ({counts_str})")
            
        except EmotionNotAvailableError as e:
            if skip_unavailable_emotions:
                if verbose:
                    print(f"  {ds_name}: Skipping (missing emotions)")
                continue
            else:
                raise
        except FileNotFoundError:
            if skip_unavailable_datasets:
                if verbose:
                    print(f"  {ds_name}: Not found, skipping")
                continue
            else:
                raise
    
    if not all_paths:
        raise ValueError(
            f"No data loaded! Check that:\n"
            f"  1. DataSets directory exists at {data_root}\n"
            f"  2. At least one dataset has the requested emotions: {emotions}"
        )
    
    if verbose:
        print("=" * 60)
        print(f"Total: {len(all_paths)} samples from {len(datasets_used)} datasets")
        print(f"Datasets used: {', '.join(datasets_used)}")
        
        # Overall distribution
        counts = {}
        for label in all_labels:
            emo = emotions[label]
            counts[emo] = counts.get(emo, 0) + 1
        print("Distribution:")
        for emo, count in sorted(counts.items(), key=lambda x: emotions.index(x[0])):
            pct = count / len(all_labels) * 100
            print(f"  {emo}: {count} ({pct:.1f}%)")
        print("=" * 60)
    
    return all_paths, all_labels, emotion_to_idx, datasets_used, all_weights


def check_emotion_availability(emotions: List[str], datasets: List[str] = None) -> Dict[str, List[str]]:
    """
    Check which datasets have which emotions.
    
    Args:
        emotions: List of emotions to check
        datasets: List of datasets to check (default: all)
    
    Returns:
        Dict mapping dataset name to list of missing emotions
    """
    if datasets is None:
        datasets = list(AVAILABLE_LOADERS.keys())
    
    results = {}
    
    for ds_name in datasets:
        loader_class = AVAILABLE_LOADERS[ds_name]
        available = set(e.lower() for e in loader_class.AVAILABLE_EMOTIONS)
        requested = set(e.lower() for e in emotions)
        missing = requested - available
        
        if missing:
            results[ds_name] = [e for e in emotions if e.lower() in missing]
    
    return results


# Convenience function
def load_4class(datasets: List[str] = None, include_songs: bool = False, **kwargs):
    """Load 4-class emotion data (Anger, Happy, Neutral, Sad)."""
    return load_datasets(
        emotions=['Anger', 'Happy', 'Neutral', 'Sad'],
        datasets=datasets,
        include_songs=include_songs,
        **kwargs
    )


def load_6class(datasets: List[str] = None, include_songs: bool = False, **kwargs):
    """Load 6-class emotion data (+ Fear, Surprise)."""
    return load_datasets(
        emotions=['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Surprise'],
        datasets=datasets,
        include_songs=include_songs,
        **kwargs
    )


if __name__ == '__main__':
    # Test the unified loader
    print("\n" + "="*60)
    print("Testing Unified Loader")
    print("="*60)
    
    # Check availability first
    print("\nChecking 6-class availability:")
    missing = check_emotion_availability(
        ['Anger', 'Happy', 'Neutral', 'Sad', 'Fear', 'Surprise']
    )
    if missing:
        for ds, emos in missing.items():
            print(f"  {ds} missing: {', '.join(emos)}")
    else:
        print("  All datasets have all 6 emotions!")
    
    # Test 4-class load
    print("\n")
    try:
        paths, labels, mapping, used = load_4class(
            datasets=['CREMA-D', 'RAVDESS', 'SAVEE', 'TESS']
        )
    except Exception as e:
        print(f"Error: {e}")

