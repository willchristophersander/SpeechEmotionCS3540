#!/usr/bin/env python3
"""
Model Performance Registry
==========================

Tracks training script outputs and their performance metrics.
Generates MODEL_PERFORMANCE.md with leaderboard and history.

Usage:
    # In training script, after saving best model:
    from scripts.performance_registry import register_model
    
    register_model(
        script_name='Train_CRNN_MultiDataset.py',
        model_path='models/crnn_multi/best.pth',
        metrics={'val_accuracy': 0.7726, 'test_accuracy': 0.75},
        config={'epochs': 150, 'batch_size': 40}
    )
    
    # To regenerate MODEL_PERFORMANCE.md:
    python scripts/performance_registry.py
    
    # To scan existing checkpoints:
    python scripts/performance_registry.py --scan
"""

MODULE_INFO = {
    'description': 'Tracks model performance across training scripts and generates MODEL_PERFORMANCE.md',
    'inputs': ['Model checkpoints with accuracy info', 'Manual registrations from training scripts'],
    'outputs': ['models/MODEL_PERFORMANCE.md', 'models/.performance_registry.json'],
    'dependencies': ['torch', 'json', 'pathlib'],
    'status': 'production'
}

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
REGISTRY_FILE = MODELS_DIR / '.performance_registry.json'
PERFORMANCE_MD = MODELS_DIR / 'MODEL_PERFORMANCE.md'


def load_registry() -> Dict[str, Any]:
    """Load the performance registry from disk."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, 'r') as f:
            return json.load(f)
    return {'models': {}, 'history': []}


def save_registry(registry: Dict[str, Any]):
    """Save the performance registry to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2, default=str)


def register_model(
    script_name: str,
    model_path: str,
    metrics: Dict[str, float],
    config: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
    datasets: Optional[List[str]] = None
):
    """
    Register a trained model and its performance.
    
    Args:
        script_name: Name of the training script (e.g., 'Train_CRNN_MultiDataset.py')
        model_path: Path to saved model (relative to project root)
        metrics: Dict with 'val_accuracy', 'test_accuracy', etc.
        config: Training configuration used
        notes: Optional notes about this run
        datasets: List of dataset names used for training (captured at training time)
    
    Note: Dataset info is captured once when the model is created and preserved
          even if the script is later updated to use different datasets.
    """
    registry = load_registry()
    
    timestamp = datetime.now().isoformat()
    
    entry = {
        'script': script_name,
        'model_path': model_path,
        'metrics': metrics,
        'config': config or {},
        'notes': notes,
        'datasets': datasets or ['UNKNOWN'],
        'timestamp': timestamp
    }
    
    # Add to history
    registry['history'].append(entry)
    
    # Track models by their actual file path (not just script)
    # This ensures each unique model file has its own record
    model_key = model_path.replace('/', '_').replace('.pth', '').replace('.pt', '')
    
    # Check if this specific model file already exists in registry
    if 'models_by_path' not in registry:
        registry['models_by_path'] = {}
    
    if model_key not in registry['models_by_path']:
        # New model - save with all info including datasets
        registry['models_by_path'][model_key] = entry
    else:
        # Existing model - only update accuracy, PRESERVE original datasets
        existing = registry['models_by_path'][model_key]
        existing_acc = existing['metrics'].get('val_accuracy', existing['metrics'].get('accuracy', 0))
        new_acc = metrics.get('val_accuracy', metrics.get('accuracy', 0))
        
        if new_acc > existing_acc:
            # Update accuracy but keep original datasets
            original_datasets = existing.get('datasets', ['UNKNOWN'])
            entry['datasets'] = original_datasets  # Preserve original
            registry['models_by_path'][model_key] = entry
    
    # Also update best model per script (for leaderboard)
    script_key = script_name.replace('.py', '')
    val_acc = metrics.get('val_accuracy', metrics.get('accuracy', 0))
    
    if script_key not in registry['models']:
        registry['models'][script_key] = entry
    else:
        existing_acc = registry['models'][script_key]['metrics'].get(
            'val_accuracy', registry['models'][script_key]['metrics'].get('accuracy', 0)
        )
        if val_acc > existing_acc:
            # For script best, also preserve datasets from the specific model
            if model_key in registry['models_by_path']:
                entry['datasets'] = registry['models_by_path'][model_key].get('datasets', ['UNKNOWN'])
            registry['models'][script_key] = entry
    
    save_registry(registry)
    datasets_str = ', '.join(entry['datasets'][:3]) + ('...' if len(entry['datasets']) > 3 else '')
    print(f"✅ Registered: {script_name} → {model_path} (acc: {val_acc*100:.2f}%, datasets: {datasets_str})")
    
    # Auto-regenerate markdown
    generate_performance_md()


def scan_checkpoints() -> List[Dict[str, Any]]:
    """Scan models folder for checkpoint files and extract metrics."""
    import torch
    
    found = []
    
    # Common checkpoint locations
    search_paths = [
        MODELS_DIR.glob('**/*.pth'),
        MODELS_DIR.glob('**/*.pt'),
        PROJECT_ROOT.glob('demo/checkpoints/**/*.pth'),
        PROJECT_ROOT.glob('CS-FAIR-DEMO/checkpoints/**/*.pth'),
    ]
    
    for glob_iter in search_paths:
        for ckpt_path in glob_iter:
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                
                if isinstance(checkpoint, dict):
                    metrics = {}
                    
                    # Extract accuracy metrics
                    for key in ['val_accuracy', 'accuracy', 'test_accuracy', 'val_acc']:
                        if key in checkpoint:
                            val = checkpoint[key]
                            if isinstance(val, (int, float)):
                                metrics['val_accuracy'] = float(val)
                                break
                    
                    # Extract epoch
                    epoch = checkpoint.get('epoch', 'unknown')
                    
                    # Extract emotions if available
                    emotions = checkpoint.get('emotions', [])
                    
                    if metrics:
                        found.append({
                            'model_path': str(ckpt_path.relative_to(PROJECT_ROOT)),
                            'metrics': metrics,
                            'epoch': epoch,
                            'emotions': emotions,
                            'script': infer_script_from_path(ckpt_path),
                            'datasets': ['UNKNOWN']  # Can't infer from old checkpoints
                        })
                        
            except Exception as e:
                pass  # Skip files that can't be loaded
    
    return found


def infer_script_from_path(model_path: Path) -> str:
    """Try to infer which script created a model based on path."""
    path_str = str(model_path).lower()
    
    if 'crnn_multi_6class' in path_str or '6class' in path_str:
        return 'Train_CRNN_MultiDataset_6Class'
    elif 'crnn_multi' in path_str:
        return 'Train_CRNN_MultiDataset'
    elif 'improved' in path_str:
        return 'Train_Improved_4Class_CRNN'
    elif 'wav2vec' in path_str:
        return 'Train_Wav2Vec2_Emotion'
    elif '4class' in path_str:
        return 'train_unified (4class)'
    elif 'feature_based' in path_str:
        return 'feature_based scripts'
    else:
        return 'Unknown'


def generate_performance_md():
    """Generate MODEL_PERFORMANCE.md from registry."""
    registry = load_registry()
    
    lines = [
        "# Model Performance Tracker",
        "",
        "Auto-generated performance tracking for trained models.",
        "",
        "## 🏆 Leaderboard (Best Models by Script)",
        "",
        "| Rank | Script | Accuracy | Datasets | Model Path |",
        "|------|--------|----------|----------|------------|",
    ]
    
    # Sort models by accuracy
    sorted_models = sorted(
        registry['models'].items(),
        key=lambda x: x[1]['metrics'].get('val_accuracy', x[1]['metrics'].get('accuracy', 0)),
        reverse=True
    )
    
    for rank, (script_key, entry) in enumerate(sorted_models, 1):
        acc = entry['metrics'].get('val_accuracy', entry['metrics'].get('accuracy', 0))
        model_path = entry.get('model_path', 'N/A')
        datasets = entry.get('datasets', ['UNKNOWN'])
        datasets_str = ', '.join(datasets[:2]) + ('...' if len(datasets) > 2 else '')
        
        medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(rank, str(rank))
        lines.append(f"| {medal} | `{script_key}` | **{acc*100:.2f}%** | {datasets_str} | `{model_path}` |")
    
    if not sorted_models:
        lines.append("| - | No models registered yet | - | - | - |")
    
    # Add breakdown by emotion classes
    lines.extend([
        "",
        "## 📊 By Classification Type",
        "",
        "### 4-Class Models (Anger, Happy, Neutral, Sad)",
        "",
        "| Script | Best Accuracy | Datasets | Notes |",
        "|--------|--------------|----------|-------|",
    ])
    
    four_class = [(k, v) for k, v in sorted_models if '6class' not in k.lower() and '6Class' not in k]
    for script_key, entry in four_class[:5]:
        acc = entry['metrics'].get('val_accuracy', entry['metrics'].get('accuracy', 0))
        notes = entry.get('notes', '-') or '-'
        datasets = entry.get('datasets', ['UNKNOWN'])
        datasets_str = ', '.join(datasets[:3]) + ('...' if len(datasets) > 3 else '')
        lines.append(f"| `{script_key}` | {acc*100:.2f}% | {datasets_str} | {notes} |")
    
    if not four_class:
        lines.append("| - | No 4-class models yet | - | - |")
    
    lines.extend([
        "",
        "### 6-Class Models (+ Fear, Surprise)",
        "",
        "| Script | Best Accuracy | Datasets | Notes |",
        "|--------|--------------|----------|-------|",
    ])
    
    six_class = [(k, v) for k, v in sorted_models if '6class' in k.lower() or '6Class' in k]
    for script_key, entry in six_class[:5]:
        acc = entry['metrics'].get('val_accuracy', entry['metrics'].get('accuracy', 0))
        notes = entry.get('notes', '-') or '-'
        datasets = entry.get('datasets', ['UNKNOWN'])
        datasets_str = ', '.join(datasets[:3]) + ('...' if len(datasets) > 3 else '')
        lines.append(f"| `{script_key}` | {acc*100:.2f}% | {datasets_str} | {notes} |")
    
    if not six_class:
        lines.append("| - | No 6-class models yet | - | - |")
    
    # Recent history
    lines.extend([
        "",
        "## 📈 Recent Training Runs",
        "",
        "| Date | Script | Accuracy | Model |",
        "|------|--------|----------|-------|",
    ])
    
    # Last 10 runs
    recent = sorted(registry['history'], key=lambda x: x.get('timestamp', ''), reverse=True)[:10]
    for entry in recent:
        timestamp = entry.get('timestamp', 'N/A')
        if timestamp != 'N/A':
            timestamp = timestamp.split('T')[0]
        script = entry.get('script', 'Unknown')
        acc = entry['metrics'].get('val_accuracy', entry['metrics'].get('accuracy', 0))
        model_path = entry.get('model_path', 'N/A')
        # Shorten path for display
        if len(model_path) > 40:
            model_path = '...' + model_path[-37:]
        lines.append(f"| {timestamp} | `{script}` | {acc*100:.2f}% | `{model_path}` |")
    
    if not recent:
        lines.append("| - | No training runs recorded | - | - |")
    
    # Usage instructions
    lines.extend([
        "",
        "---",
        "",
        "## How to Register Models",
        "",
        "Add this to your training script after saving the best model:",
        "",
        "```python",
        "from scripts.performance_registry import register_model",
        "",
        "# Define datasets AT TRAINING TIME - this info is preserved forever",
        "TRAINING_DATASETS = ['CREMA-D', 'RAVDESS', 'SAVEE', 'TESS']",
        "",
        "register_model(",
        "    script_name='YourScript.py',",
        "    model_path='models/your_model.pth',",
        "    metrics={'val_accuracy': 0.77, 'test_accuracy': 0.75},",
        "    notes='Brief description of this run',",
        "    datasets=TRAINING_DATASETS  # Captured once, never overwritten",
        ")",
        "```",
        "",
        "**Note:** Dataset info is captured when the model is first created.",
        "Even if the script is later modified to use different datasets,",
        "the original dataset info for existing models is preserved.",
        "",
        f"*Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
    ])
    
    # Write file
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PERFORMANCE_MD, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"✅ Generated {PERFORMANCE_MD.relative_to(PROJECT_ROOT)}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Model Performance Registry")
    print("=" * 60)
    
    if '--scan' in sys.argv:
        print("\nScanning for checkpoints...")
        found = scan_checkpoints()
        
        if found:
            print(f"\nFound {len(found)} checkpoints with metrics:")
            for item in found:
                acc = item['metrics'].get('val_accuracy', 0)
                print(f"  • {item['model_path']}: {acc*100:.2f}% ({item['script']})")
            
            print("\nRegistering found models...")
            for item in found:
                register_model(
                    script_name=item['script'],
                    model_path=item['model_path'],
                    metrics=item['metrics'],
                    notes=f"Auto-scanned. Epoch: {item.get('epoch', '?')}"
                )
        else:
            print("No checkpoints with metrics found.")
    else:
        # Just regenerate the markdown
        print("\nRegenerating MODEL_PERFORMANCE.md...")
        generate_performance_md()
    
    print("\nDone!")


if __name__ == '__main__':
    main()

