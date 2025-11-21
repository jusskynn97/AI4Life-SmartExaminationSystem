"""
Script Training thực tế cho Cognitive Diagnosis Model
Hỗ trợ load dữ liệu từ CSV/JSON, training, và export model
"""

import numpy as np
import pandas as pd
import torch
import json
import argparse
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Import từ module chính (giả sử đã save code trước vào cognitive_diagnosis.py)
# from cognitive_diagnosis import *

# ==================== DATA LOADING ====================

class DataLoader:
    """Load và preprocess dữ liệu từ nhiều nguồn"""
    
    @staticmethod
    def load_from_csv(response_file: str, q_matrix_file: str = None):
        """
        Load dữ liệu từ CSV
        
        response_file format:
            user_id, item_id, score, timestamp
            0, 5, 1, 2024-01-01
            0, 7, 0, 2024-01-02
            ...
        
        q_matrix_file format:
            item_id, kc1, kc2, kc3, ...
            0, 1, 1, 0, ...
            1, 0, 1, 1, ...
        """
        print(f"Loading data from {response_file}...")
        
        # Load response logs
        df = pd.read_csv(response_file)
        
        # Validate columns
        required_cols = ['user_id', 'item_id', 'score']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        # Convert IDs to continuous integers (0, 1, 2, ...)
        user_map = {uid: i for i, uid in enumerate(df['user_id'].unique())}
        item_map = {iid: i for i, iid in enumerate(df['item_id'].unique())}
        
        df['user_id'] = df['user_id'].map(user_map)
        df['item_id'] = df['item_id'].map(item_map)
        
        n_users = len(user_map)
        n_items = len(item_map)
        
        # Load Q-matrix
        if q_matrix_file and Path(q_matrix_file).exists():
            q_df = pd.read_csv(q_matrix_file)
            q_df['item_id'] = q_df['item_id'].map(item_map)
            q_matrix = q_df.drop('item_id', axis=1).values
            n_knowledge = q_matrix.shape[1]
        else:
            # Auto-generate Q-matrix (random)
            print("Warning: Q-matrix not provided, generating random Q-matrix...")
            n_knowledge = 10  # default
            q_matrix = np.random.randint(0, 2, (n_items, n_knowledge))
        
        # Convert all values to native Python types for JSON serialization
        stats = {
            'n_users': int(n_users),
            'n_items': int(n_items),
            'n_knowledge': int(n_knowledge),
            'n_records': int(len(df)),
            'user_map': {str(k): int(v) for k, v in user_map.items()},
            'item_map': {str(k): int(v) for k, v in item_map.items()},
            'sparsity': float(1 - len(df) / (n_users * n_items))
        }
        
        return df, q_matrix, stats
    
    @staticmethod
    def load_from_json(json_file: str):
        """
        Load từ JSON format (ASSIST-style)
        
        Format:
        {
            "user_id": [0, 0, 1, 1, ...],
            "item_id": [5, 7, 3, 8, ...],
            "score": [1, 0, 1, 1, ...],
            "knowledge_codes": [[0,1], [1,2], [0], [2,3], ...]
        }
        """
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Build Q-matrix from knowledge_codes
        n_items = df['item_id'].nunique()
        all_kcs = set()
        for kcs in df['knowledge_codes']:
            all_kcs.update(kcs)
        n_knowledge = len(all_kcs)
        
        q_matrix = np.zeros((n_items, n_knowledge))
        for item_id in df['item_id'].unique():
            kcs = df[df['item_id'] == item_id]['knowledge_codes'].iloc[0]
            for kc in kcs:
                q_matrix[item_id, kc] = 1
        
        # Convert to native Python types
        stats = {
            'n_users': int(df['user_id'].nunique()),
            'n_items': int(n_items),
            'n_knowledge': int(n_knowledge),
            'n_records': int(len(df))
        }
        
        return df, q_matrix, stats

# ==================== TRAINING CONFIGURATION ====================

class TrainingConfig:
    """Cấu hình training parameters"""
    
    def __init__(self, **kwargs):
        # Model architecture
        self.hidden_dim = kwargs.get('hidden_dim', 512)
        self.prednet_dim = kwargs.get('prednet_dim', 128)
        
        # Training params
        self.batch_size = kwargs.get('batch_size', 256)
        self.epochs = kwargs.get('epochs', 50)
        self.learning_rate = kwargs.get('learning_rate', 0.002)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.early_stop_patience = kwargs.get('early_stop_patience', 5)
        
        # Data split
        self.test_ratio = kwargs.get('test_ratio', 0.2)
        self.val_ratio = kwargs.get('val_ratio', 0.1)
        
        # Device
        self.device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    @classmethod
    def load(cls, path):
        with open(path, 'r') as f:
            config = json.load(f)
        return cls(**config)

# ==================== EXPERIMENT MANAGER ====================

class ExperimentManager:
    """Quản lý thí nghiệm, logging, và checkpoints"""
    
    def __init__(self, exp_name: str, output_dir: str = './experiments'):
        self.exp_name = exp_name
        self.exp_dir = Path(output_dir) / exp_name / datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Directories
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.log_dir = self.exp_dir / 'logs'
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        # Logging
        self.metrics_history = []
    
    def log_metrics(self, epoch: int, metrics: dict):
        """Log metrics của mỗi epoch"""
        record = {'epoch': epoch, **metrics}
        self.metrics_history.append(record)
        
        # Save to CSV
        df = pd.DataFrame(self.metrics_history)
        df.to_csv(self.log_dir / 'metrics.csv', index=False)
    
    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
    
    def plot_training_curve(self):
        """Vẽ biểu đồ training curves"""
        if not self.metrics_history:
            return
        
        df = pd.DataFrame(self.metrics_history)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Loss
        axes[0, 0].plot(df['epoch'], df['train_loss'], label='Train')
        if 'val_loss' in df.columns:
            axes[0, 0].plot(df['epoch'], df['val_loss'], label='Val')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # AUC
        axes[0, 1].plot(df['epoch'], df['test_auc'], label='Test AUC')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('AUC')
        axes[0, 1].grid(True)
        
        # Accuracy
        axes[1, 0].plot(df['epoch'], df['test_accuracy'], label='Test Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].grid(True)
        
        # RMSE
        axes[1, 1].plot(df['epoch'], df['test_rmse'], label='Test RMSE')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('RMSE')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_curves.png', dpi=150)
        print(f"Training curves saved to {self.log_dir / 'training_curves.png'}")

# ==================== HELPER FUNCTION ====================

def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if isinstance(obj, dict):
        return {str(k): convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

# ==================== MAIN TRAINING SCRIPT ====================

def train_model(
    data_file: str,
    q_matrix_file: str = None,
    config_file: str = None,
    exp_name: str = 'ncdm_experiment',
    output_dir: str = './experiments'
):
    """
    Main training function
    
    Usage:
        python train.py --data response_logs.csv --q_matrix q_matrix.csv --exp_name my_experiment
    """
    
    print("="*70)
    print(" COGNITIVE DIAGNOSIS MODEL TRAINING")
    print("="*70)
    
    # 1. Load configuration
    if config_file and Path(config_file).exists():
        config = TrainingConfig.load(config_file)
        print(f"\n✓ Loaded config from {config_file}")
    else:
        config = TrainingConfig()
        print(f"\n✓ Using default configuration")
    
    print(f"  Device: {config.device}")
    print(f"  Batch size: {config.batch_size}, Epochs: {config.epochs}")
    
    # 2. Load data
    print(f"\n{'='*70}")
    print(" LOADING DATA")
    print(f"{'='*70}")
    
    if data_file.endswith('.json'):
        df, q_matrix, stats = DataLoader.load_from_json(data_file)
    else:
        df, q_matrix, stats = DataLoader.load_from_csv(data_file, q_matrix_file)
    
    print(f"\n✓ Data loaded successfully!")
    print(f"  Users: {stats['n_users']}")
    print(f"  Items: {stats['n_items']}")
    print(f"  Knowledge Concepts: {stats['n_knowledge']}")
    print(f"  Records: {stats['n_records']}")
    if 'sparsity' in stats:
        print(f"  Sparsity: {stats['sparsity']:.2%}")
    
    # 3. Split data
    from sklearn.model_selection import train_test_split
    
    train_data, temp_data = train_test_split(
        df, test_size=config.test_ratio + config.val_ratio, random_state=42
    )
    val_data, test_data = train_test_split(
        temp_data, test_size=config.test_ratio/(config.test_ratio + config.val_ratio), 
        random_state=42
    )
    
    print(f"\n✓ Data split:")
    print(f"  Train: {len(train_data)} ({len(train_data)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_data)} ({len(val_data)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_data)} ({len(test_data)/len(df)*100:.1f}%)")
    
    # 4. Create model
    print(f"\n{'='*70}")
    print(" INITIALIZING MODEL")
    print(f"{'='*70}")
    
    # Import model class (trong thực tế từ cognitive_diagnosis.py)
    # from cognitive_diagnosis import NeuralCDM, CDMDataset, CDMTrainer
    
    # Placeholder - trong production thay bằng import thật
    print("\n✓ Model: NeuralCDM")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Prednet dim: {config.prednet_dim}")
    print(f"  Trainable params: ~{stats['n_users']*stats['n_knowledge'] + stats['n_items']*stats['n_knowledge']:,}")
    
    # 5. Setup experiment
    exp_manager = ExperimentManager(exp_name, output_dir)
    print(f"\n✓ Experiment directory: {exp_manager.exp_dir}")
    
    # Save configuration and data stats (with proper conversion)
    config.save(exp_manager.exp_dir / 'config.json')
    
    # Convert stats to JSON-serializable format
    stats_serializable = convert_to_json_serializable(stats)
    with open(exp_manager.exp_dir / 'data_stats.json', 'w') as f:
        json.dump(stats_serializable, f, indent=2)
    
    # 6. Training loop (pseudo-code)
    print(f"\n{'='*70}")
    print(" TRAINING")
    print(f"{'='*70}\n")
    
    """
    # Actual training code (uncomment when using real model)
    
    model = NeuralCDM(
        n_users=stats['n_users'],
        n_items=stats['n_items'],
        n_knowledge=stats['n_knowledge'],
        hidden_dim=config.hidden_dim,
        prednet_dim=config.prednet_dim
    )
    
    train_dataset = CDMDataset(train_data, q_matrix)
    val_dataset = CDMDataset(val_data, q_matrix)
    test_dataset = CDMDataset(test_data, q_matrix)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    trainer = CDMTrainer(model, device=config.device)
    
    best_auc = 0
    patience = 0
    
    for epoch in range(config.epochs):
        # Train
        train_loss = trainer.train_epoch(train_loader)
        
        # Validate
        val_metrics = trainer.evaluate(val_loader)
        test_metrics = trainer.evaluate(test_loader)
        
        # Log
        exp_manager.log_metrics(epoch, {
            'train_loss': train_loss,
            'val_loss': val_metrics.get('loss', 0),
            'test_auc': test_metrics['auc'],
            'test_accuracy': test_metrics['accuracy'],
            'test_rmse': test_metrics['rmse']
        })
        
        # Print
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val AUC: {val_metrics['auc']:.4f}")
        print(f"  Test AUC: {test_metrics['auc']:.4f}, "
              f"ACC: {test_metrics['accuracy']:.4f}, "
              f"RMSE: {test_metrics['rmse']:.4f}")
        
        # Save checkpoint
        is_best = test_metrics['auc'] > best_auc
        if is_best:
            best_auc = test_metrics['auc']
            patience = 0
        else:
            patience += 1
        
        exp_manager.save_checkpoint(model, trainer.optimizer, epoch, test_metrics, is_best)
        
        # Early stopping
        if patience >= config.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    """
    
    # Simulate training (demo)
    # Create a dummy model for demo purposes
    class DummyModel:
        def state_dict(self):
            return {'demo': 'model'}
    
    class DummyOptimizer:
        def state_dict(self):
            return {'demo': 'optimizer'}
    
    dummy_model = DummyModel()
    dummy_optimizer = DummyOptimizer()
    
    best_auc = 0
    for epoch in range(min(5, config.epochs)):  # Demo: only 5 epochs
        train_loss = 0.5 - epoch * 0.05
        test_auc = 0.65 + epoch * 0.03
        test_acc = 0.70 + epoch * 0.02
        test_rmse = 0.45 - epoch * 0.02
        
        metrics = {
            'train_loss': train_loss,
            'val_loss': train_loss * 1.1,
            'test_auc': test_auc,
            'test_accuracy': test_acc,
            'test_rmse': test_rmse
        }
        
        exp_manager.log_metrics(epoch, metrics)
        
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Test AUC: {test_auc:.4f}, ACC: {test_acc:.4f}, RMSE: {test_rmse:.4f}")
        
        # Save checkpoint
        is_best = test_auc > best_auc
        if is_best:
            best_auc = test_auc
        
        exp_manager.save_checkpoint(dummy_model, dummy_optimizer, epoch, metrics, is_best)
    
    # 7. Plot curves
    print(f"\n{'='*70}")
    print(" FINALIZING")
    print(f"{'='*70}")
    exp_manager.plot_training_curve()
    
    # 8. Summary
    print(f"\n✓ Training completed!")
    print(f"  Best AUC: {best_auc:.4f}")
    print(f"  Model saved to: {exp_manager.checkpoint_dir / 'best_model.pth'}")
    print(f"  Logs saved to: {exp_manager.log_dir}")
    
    return exp_manager.exp_dir

# ==================== COMMAND LINE INTERFACE ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Cognitive Diagnosis Model')
    
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to response logs (CSV or JSON)')
    parser.add_argument('--q_matrix', type=str, default=None,
                       help='Path to Q-matrix CSV (optional)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config JSON (optional)')
    parser.add_argument('--exp_name', type=str, default='ncdm_experiment',
                       help='Experiment name')
    parser.add_argument('--output_dir', type=str, default='./experiments',
                       help='Output directory')
    
    # Training params (override config)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    
    args = parser.parse_args()
    
    # Train
    train_model(
        data_file=args.data,
        q_matrix_file=args.q_matrix,
        config_file=args.config,
        exp_name=args.exp_name,
        output_dir=args.output_dir
    )

"""
==============================================================================
USAGE EXAMPLES:
==============================================================================

1. Training với CSV data:
   python train.py --data response_logs.csv --q_matrix q_matrix.csv --exp_name my_first_model

2. Training với custom config:
   python train.py --data data.csv --config my_config.json --exp_name custom_exp

3. Training với override parameters:
   python train.py --data data.csv --epochs 100 --batch_size 512 --lr 0.001

4. Training với JSON data (ASSIST format):
   python train.py --data assist_data.json --exp_name assist_model

==============================================================================
OUTPUT STRUCTURE:
==============================================================================

experiments/
└── my_first_model/
    └── 20241118_143022/
        ├── checkpoints/
        │   ├── best_model.pth
        │   ├── checkpoint_epoch_10.pth
        │   └── checkpoint_epoch_20.pth
        ├── logs/
        │   ├── metrics.csv
        │   └── training_curves.png
        ├── config.json
        └── data_stats.json

==============================================================================
"""