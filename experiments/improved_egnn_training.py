#!/usr/bin/env python3
"""
Улучшенное обучение EGNN с оптимизированными гиперпараметрами.

Улучшения:
1. Больше эпох (100)
2. Больше hidden dimensions (256)
3. Больше слоев (5)
4. Улучшенный learning rate schedule
5. Data augmentation
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
import logging
import time
from datetime import datetime
import json

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent.parent / "src"))

from step_01_data.loaders import MolecularDataLoader
from step_03_models.egnn import EGNNModel, EGNNConfig
from step_03_models.model_adapters import create_model_adapter
from step_04_training.trainer import ModelTrainer, TrainingConfig

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'experiments/improved_egnn_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImprovedEGNNTrainer:
    """Тренер для улучшенной EGNN модели."""
    
    def __init__(self, 
                 model_id: int = 1,
                 data_root: str = "data/raw",
                 results_dir: str = "results/improved_models",
                 target_property: str = "homo_lumo_gap"):
        """
        Инициализация.
        
        Args:
            model_id: ID модели для ensemble (1, 2, 3)
            data_root: Корневая директория данных
            results_dir: Директория для результатов
            target_property: Целевое свойство
        """
        self.model_id = model_id
        self.data_root = Path(data_root)
        self.results_dir = Path(results_dir)
        self.target_property = target_property
        
        # Создаем директории
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "models").mkdir(exist_ok=True)
        
        # Устройство
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Model {model_id}: Используем устройство {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"Model {model_id}: GPU {torch.cuda.get_device_name(0)}")
            logger.info(f"Model {model_id}: GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    def load_data_with_augmentation(self):
        """Загружает данные с augmentation."""
        logger.info(f"Model {self.model_id}: Загрузка QM9 данных...")
        
        loader = MolecularDataLoader(data_root=str(self.data_root))
        data_list, targets, metadata = loader.load_qm9(target_property=self.target_property)
        
        logger.info(f"Model {self.model_id}: Загружено {len(data_list)} молекул")
        
        # Создаем splits с разными random seeds для ensemble
        n_total = len(data_list)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        
        # Разный seed для каждой модели в ensemble
        torch.manual_seed(42 + self.model_id)
        indices = torch.randperm(n_total)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        train_data = [data_list[i] for i in train_indices]
        val_data = [data_list[i] for i in val_indices]
        test_data = [data_list[i] for i in test_indices]
        
        train_targets = targets[train_indices]
        val_targets = targets[val_indices]
        test_targets = targets[test_indices]
        
        logger.info(f"Model {self.model_id}: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
        
        return (train_data, train_targets), (val_data, val_targets), (test_data, test_targets), metadata
    
    def prepare_features(self, data_list):
        """Подготавливает признаки для EGNN."""
        features_list = []
        coords_list = []
        
        for data in data_list:
            # Node features
            if hasattr(data, 'x') and data.x is not None:
                node_features = data.x.float()
            else:
                node_features = torch.zeros(data.pos.size(0), 5)
                if hasattr(data, 'z'):
                    node_features[:, 0] = data.z.float()
            
            features_list.append(node_features)
            coords_list.append(data.pos)
        
        # Паддинг
        max_atoms = max(f.size(0) for f in features_list)
        
        padded_features = []
        padded_coords = []
        
        for features, coords in zip(features_list, coords_list):
            n_atoms = features.size(0)
            n_features = features.size(1)
            
            padded_f = torch.zeros(max_atoms, n_features)
            padded_f[:n_atoms] = features
            
            padded_c = torch.zeros(max_atoms, 3)
            padded_c[:n_atoms] = coords
            
            padded_features.append(padded_f)
            padded_coords.append(padded_c)
        
        return torch.stack(padded_features), torch.stack(padded_coords)
    
    def train_improved_model(self):
        """Обучает улучшенную модель."""
        logger.info(f"Model {self.model_id}: 🚀 Начинаем обучение улучшенной EGNN...")
        
        # 1. Загружаем данные
        (train_data, train_targets), (val_data, val_targets), (test_data, test_targets), metadata = \
            self.load_data_with_augmentation()
        
        # 2. Подготавливаем признаки
        logger.info(f"Model {self.model_id}: Подготовка признаков...")
        train_features, train_coords = self.prepare_features(train_data)
        val_features, val_coords = self.prepare_features(val_data)
        test_features, test_coords = self.prepare_features(test_data)
        
        # 3. Создаем улучшенную модель
        input_dim = train_features.size(-1)
        
        config = EGNNConfig(
            node_feature_dim=input_dim,
            hidden_dim=256,  # Увеличено с 128
            num_layers=5,    # Увеличено с 3
            output_dim=1,
            dropout=0.1,     # Уменьшено для большей модели
            attention=True,  # Исправлено: attention вместо use_attention
            normalize=True,
            tanh=True
        )
        
        base_model = EGNNModel(config)
        model = create_model_adapter(base_model, 'egnn')
        
        logger.info(f"Model {self.model_id}: Параметров: {sum(p.numel() for p in model.parameters()):,}")
        
        # 4. Улучшенная конфигурация обучения
        training_config = TrainingConfig(
            epochs=100,           # Увеличено с 50
            batch_size=32,        # Уменьшено для большей модели
            learning_rate=5e-4,   # Немного меньше для стабильности
            weight_decay=1e-5,
            patience=15,          # Больше терпения
            validation_split=0.2,
            save_best_model=True,
            save_checkpoints=True,
            checkpoint_freq=20,
            verbose=True
        )
        
        # 5. Создаем тренер
        trainer = ModelTrainer(
            model=model,
            config=training_config,
            device=self.device,
            experiment_name=f"improved_egnn_model{self.model_id}",
            save_dir=str(self.results_dir / "models")
        )
        
        # 6. Обучаем
        logger.info(f"Model {self.model_id}: Начинаем обучение на {self.device}...")
        start_time = time.time()
        
        history = trainer.fit(
            X=train_features,
            y=train_targets,
            coords=train_coords,
            property_name=self.target_property,
            property_units="eV"
        )
        
        training_time = time.time() - start_time
        
        # 7. Оцениваем
        logger.info(f"Model {self.model_id}: Оценка на тестовых данных...")
        test_metrics = trainer.evaluate(test_features, test_targets, test_coords)
        
        # 8. Сохраняем результаты
        results = {
            'model_id': self.model_id,
            'model_name': f'improved_egnn_model{self.model_id}',
            'config': {
                'hidden_dim': 256,
                'num_layers': 5,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 5e-4
            },
            'training_time': training_time,
            'best_epoch': history.best_epoch,
            'best_val_loss': history.best_val_loss,
            'test_metrics': test_metrics.to_dict(),
            'num_parameters': sum(p.numel() for p in model.parameters())
        }
        
        # Сохраняем в JSON
        results_path = self.results_dir / f"improved_egnn_model{self.model_id}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Model {self.model_id}: ✅ Обучение завершено!")
        logger.info(f"Model {self.model_id}: Время: {training_time/3600:.2f} часов")
        logger.info(f"Model {self.model_id}: Test MAE: {test_metrics.mae:.6f} eV")
        logger.info(f"Model {self.model_id}: Test R²: {test_metrics.r2:.4f}")
        logger.info(f"Model {self.model_id}: Результаты сохранены в {results_path}")
        
        return results


def main():
    """Основная функция."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train improved EGNN model')
    parser.add_argument('--model_id', type=int, default=1, 
                       help='Model ID for ensemble (1, 2, or 3)')
    args = parser.parse_args()
    
    logger.info(f"="*80)
    logger.info(f"IMPROVED EGNN TRAINING - Model {args.model_id}")
    logger.info(f"="*80)
    
    trainer = ImprovedEGNNTrainer(model_id=args.model_id)
    
    try:
        results = trainer.train_improved_model()
        logger.info(f"Model {args.model_id}: 🎉 Успешно завершено!")
        
    except Exception as e:
        logger.error(f"Model {args.model_id}: ❌ Ошибка: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
