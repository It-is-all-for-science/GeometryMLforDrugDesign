"""
Система обучения моделей для молекулярного машинного обучения.

Содержит класс ModelTrainer с общим интерфейсом для обучения
различных типов моделей (EGNN, baseline модели).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
import time
import os
from pathlib import Path
import json
import pickle

from .metrics import MetricsCalculator, ModelMetrics
from .utils import EarlyStopping, LearningRateScheduler, CheckpointManager

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Конфигурация для обучения модели."""
    
    # Основные параметры
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    
    # Learning rate scheduling
    lr_scheduler: str = "plateau"  # "plateau", "cosine", "step", "none"
    lr_factor: float = 0.5
    lr_patience: int = 5
    
    # Валидация
    validation_split: float = 0.2
    validation_freq: int = 1  # Каждые N эпох
    
    # Сохранение
    save_best_model: bool = True
    save_checkpoints: bool = True
    checkpoint_freq: int = 10  # Каждые N эпох
    
    # Логирование
    log_freq: int = 10  # Каждые N батчей
    verbose: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует конфигурацию в словарь."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'lr_scheduler': self.lr_scheduler,
            'lr_factor': self.lr_factor,
            'lr_patience': self.lr_patience,
            'validation_split': self.validation_split,
            'validation_freq': self.validation_freq,
            'save_best_model': self.save_best_model,
            'save_checkpoints': self.save_checkpoints,
            'checkpoint_freq': self.checkpoint_freq,
            'log_freq': self.log_freq,
            'verbose': self.verbose
        }


@dataclass
class TrainingHistory:
    """История обучения модели."""
    
    train_losses: List[float]
    val_losses: List[float]
    train_metrics: List[Dict[str, float]]
    val_metrics: List[Dict[str, float]]
    learning_rates: List[float]
    epochs_completed: int
    best_epoch: int
    best_val_loss: float
    training_time: float
    
    def save(self, path: str):
        """Сохраняет историю обучения."""
        def convert_to_python_types(obj):
            """Конвертирует numpy типы в стандартные Python типы."""
            if hasattr(obj, 'item'):  # numpy scalar
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, list):
                return [convert_to_python_types(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_python_types(value) for key, value in obj.items()}
            else:
                return obj
        
        data = {
            'train_losses': convert_to_python_types(self.train_losses),
            'val_losses': convert_to_python_types(self.val_losses),
            'train_metrics': convert_to_python_types(self.train_metrics),
            'val_metrics': convert_to_python_types(self.val_metrics),
            'learning_rates': convert_to_python_types(self.learning_rates),
            'epochs_completed': self.epochs_completed,
            'best_epoch': self.best_epoch,
            'best_val_loss': convert_to_python_types(self.best_val_loss),
            'training_time': self.training_time
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingHistory':
        """Загружает историю обучения."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(**data)


class ModelTrainer:
    """
    Универсальный тренер для моделей молекулярного ML.
    
    Поддерживает обучение EGNN, baseline моделей и других архитектур
    с единым интерфейсом и продвинутыми возможностями.
    """
    
    def __init__(self,
                 model: nn.Module,
                 config: TrainingConfig,
                 device: Optional[torch.device] = None,
                 experiment_name: str = "experiment",
                 save_dir: str = "results/models"):
        """
        Инициализация тренера.
        
        Args:
            model: Модель для обучения
            config: Конфигурация обучения
            device: Устройство для вычислений
            experiment_name: Название эксперимента
            save_dir: Директория для сохранения результатов
        """
        self.model = model
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir) / experiment_name
        
        # Создаем директории
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Перемещаем модель на устройство
        self.model.to(self.device)
        
        # Инициализируем компоненты
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.lr_scheduler = None
        self.early_stopping = None
        self.checkpoint_manager = None
        self.metrics_calculator = None
        
        # История обучения
        self.history = None
        
        logger.info(f"Инициализирован ModelTrainer для {experiment_name}")
        logger.info(f"Устройство: {self.device}")
        logger.info(f"Параметров модели: {sum(p.numel() for p in model.parameters()):,}")
    
    def setup_training(self,
                      property_name: str = "molecular_property",
                      property_units: str = "eV"):
        """
        Настраивает компоненты для обучения.
        
        Args:
            property_name: Название предсказываемого свойства
            property_units: Единицы измерения
        """
        # Оптимизатор
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        if self.config.lr_scheduler == "plateau":
            self.lr_scheduler = LearningRateScheduler(
                self.optimizer,
                mode='min',
                factor=self.config.lr_factor,
                patience=self.config.lr_patience,
                verbose=self.config.verbose
            )
        elif self.config.lr_scheduler == "cosine":
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.lr_scheduler == "step":
            self.lr_scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=self.config.lr_factor
            )
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.patience,
            min_delta=self.config.min_delta,
            verbose=self.config.verbose
        )
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            save_dir=self.save_dir,
            save_best=self.config.save_best_model,
            save_checkpoints=self.config.save_checkpoints
        )
        
        # Metrics calculator
        self.metrics_calculator = MetricsCalculator(
            property_name=property_name,
            property_units=property_units
        )
        
        # Сохраняем конфигурацию
        config_path = self.save_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        logger.info("Настройка обучения завершена")
    
    def create_data_loaders(self,
                           X: torch.Tensor,
                           y: torch.Tensor,
                           coords: Optional[torch.Tensor] = None) -> Tuple[DataLoader, DataLoader]:
        """
        Создает DataLoader'ы для обучения и валидации.
        
        Args:
            X: Признаки [N, features]
            y: Целевые значения [N, 1] или [N]
            coords: Координаты атомов [N, atoms, 3] (для EGNN)
        
        Returns:
            Tuple[DataLoader, DataLoader]: train_loader, val_loader
        """
        # Проверяем размерности
        n_samples = X.shape[0]
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        
        # Разделяем на train/val
        n_val = int(n_samples * self.config.validation_split)
        n_train = n_samples - n_val
        
        # Создаем случайные индексы
        indices = torch.randperm(n_samples)
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]
        
        # Разделяем данные
        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]
        
        # Создаем datasets
        if coords is not None:
            coords_train, coords_val = coords[train_indices], coords[val_indices]
            train_dataset = TensorDataset(X_train, coords_train, y_train)
            val_dataset = TensorDataset(X_val, coords_val, y_val)
        else:
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
        
        # Создаем data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False
        )
        
        logger.info(f"Создали DataLoader'ы: train={len(train_dataset)}, val={len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Обучает модель на одной эпохе.
        
        Args:
            train_loader: DataLoader для обучения
        
        Returns:
            Tuple[float, Dict[str, float]]: средний loss и метрики
        """
        self.model.train()
        
        epoch_losses = []
        all_predictions = []
        all_targets = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Распаковываем batch
            if len(batch) == 3:  # X, coords, y (для EGNN)
                X, coords, y = batch
                X, coords, y = X.to(self.device), coords.to(self.device), y.to(self.device)
                batch_data = (X, coords)
            else:  # X, y (для baseline моделей)
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)
                batch_data = X
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if isinstance(batch_data, tuple):
                predictions = self.model(*batch_data)
            else:
                predictions = self.model(batch_data)
            
            # Вычисляем loss
            loss = self.criterion(predictions, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Сохраняем результаты
            epoch_losses.append(loss.item())
            all_predictions.append(predictions.detach().cpu())
            all_targets.append(y.detach().cpu())
            
            # Логирование
            if self.config.verbose and batch_idx % self.config.log_freq == 0:
                logger.debug(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.6f}")
        
        # Вычисляем метрики для эпохи
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics_calculator.calculate_metrics(all_targets, all_predictions)
        
        avg_loss = np.mean(epoch_losses)
        
        return avg_loss, metrics.to_dict()
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """
        Валидирует модель на одной эпохе.
        
        Args:
            val_loader: DataLoader для валидации
        
        Returns:
            Tuple[float, Dict[str, float]]: средний loss и метрики
        """
        self.model.eval()
        
        epoch_losses = []
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Распаковываем batch
                if len(batch) == 3:  # X, coords, y (для EGNN)
                    X, coords, y = batch
                    X, coords, y = X.to(self.device), coords.to(self.device), y.to(self.device)
                    batch_data = (X, coords)
                else:  # X, y (для baseline моделей)
                    X, y = batch
                    X, y = X.to(self.device), y.to(self.device)
                    batch_data = X
                
                # Forward pass
                if isinstance(batch_data, tuple):
                    predictions = self.model(*batch_data)
                else:
                    predictions = self.model(batch_data)
                
                # Вычисляем loss
                loss = self.criterion(predictions, y)
                
                # Сохраняем результаты
                epoch_losses.append(loss.item())
                all_predictions.append(predictions.cpu())
                all_targets.append(y.cpu())
        
        # Вычисляем метрики для эпохи
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics = self.metrics_calculator.calculate_metrics(all_targets, all_predictions)
        
        avg_loss = np.mean(epoch_losses)
        
        return avg_loss, metrics.to_dict()
    
    def fit(self,
            X: torch.Tensor,
            y: torch.Tensor,
            coords: Optional[torch.Tensor] = None,
            property_name: str = "molecular_property",
            property_units: str = "eV") -> TrainingHistory:
        """
        Обучает модель на данных.
        
        Args:
            X: Признаки [N, features]
            y: Целевые значения [N, 1] или [N]
            coords: Координаты атомов [N, atoms, 3] (для EGNN)
            property_name: Название свойства
            property_units: Единицы измерения
        
        Returns:
            TrainingHistory: История обучения
        """
        logger.info(f"Начинаем обучение модели {self.experiment_name}")
        start_time = time.time()
        
        # Настраиваем обучение
        self.setup_training(property_name, property_units)
        
        # Создаем data loaders
        train_loader, val_loader = self.create_data_loaders(X, y, coords)
        
        # Инициализируем историю
        self.history = TrainingHistory(
            train_losses=[],
            val_losses=[],
            train_metrics=[],
            val_metrics=[],
            learning_rates=[],
            epochs_completed=0,
            best_epoch=0,
            best_val_loss=float('inf'),
            training_time=0.0
        )
        
        # Основной цикл обучения
        for epoch in range(self.config.epochs):
            epoch_start_time = time.time()
            
            # Обучение
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Валидация
            val_loss, val_metrics = None, None
            if epoch % self.config.validation_freq == 0:
                val_loss, val_metrics = self.validate_epoch(val_loader)
            
            # Обновляем learning rate
            if self.lr_scheduler is not None:
                if isinstance(self.lr_scheduler, LearningRateScheduler):
                    self.lr_scheduler.step(val_loss if val_loss is not None else train_loss)
                else:
                    self.lr_scheduler.step()
            
            # Сохраняем историю
            self.history.train_losses.append(train_loss)
            if val_loss is not None:
                self.history.val_losses.append(val_loss)
            self.history.train_metrics.append(train_metrics)
            if val_metrics is not None:
                self.history.val_metrics.append(val_metrics)
            self.history.learning_rates.append(self.optimizer.param_groups[0]['lr'])
            self.history.epochs_completed = epoch + 1
            
            # Проверяем на лучшую модель
            current_val_loss = val_loss if val_loss is not None else train_loss
            if current_val_loss < self.history.best_val_loss:
                self.history.best_val_loss = current_val_loss
                self.history.best_epoch = epoch
                
                # Сохраняем лучшую модель
                if self.config.save_best_model:
                    self.checkpoint_manager.save_best_model(
                        self.model, self.optimizer, epoch, current_val_loss
                    )
            
            # Сохраняем checkpoint
            if self.config.save_checkpoints and epoch % self.config.checkpoint_freq == 0:
                self.checkpoint_manager.save_checkpoint(
                    self.model, self.optimizer, epoch, current_val_loss
                )
            
            # Early stopping
            if self.early_stopping.should_stop(current_val_loss):
                logger.info(f"Early stopping на эпохе {epoch}")
                break
            
            # Логирование
            if self.config.verbose:
                epoch_time = time.time() - epoch_start_time
                log_msg = f"Epoch {epoch+1}/{self.config.epochs} ({epoch_time:.1f}s) - "
                log_msg += f"Train Loss: {train_loss:.6f}"
                if val_loss is not None:
                    log_msg += f", Val Loss: {val_loss:.6f}"
                    log_msg += f", Val MAE: {val_metrics['mae']:.6f}"
                    log_msg += f", Val R²: {val_metrics['r2']:.4f}"
                log_msg += f", LR: {self.optimizer.param_groups[0]['lr']:.2e}"
                logger.info(log_msg)
        
        # Финализируем обучение
        total_time = time.time() - start_time
        self.history.training_time = total_time
        
        # Сохраняем историю
        history_path = self.save_dir / "training_history.json"
        self.history.save(str(history_path))
        
        logger.info(f"Обучение завершено за {total_time:.1f}s")
        logger.info(f"Лучшая эпоха: {self.history.best_epoch}, Val Loss: {self.history.best_val_loss:.6f}")
        
        return self.history
    
    def predict(self, X: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Делает предсказания на новых данных.
        
        Args:
            X: Признаки [N, features]
            coords: Координаты атомов [N, atoms, 3] (для EGNN)
        
        Returns:
            torch.Tensor: Предсказания [N, 1]
        """
        self.model.eval()
        
        predictions = []
        
        # Создаем DataLoader для предсказаний
        if coords is not None:
            dataset = TensorDataset(X, coords)
        else:
            dataset = TensorDataset(X)
        
        data_loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False
        )
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 2 and coords is not None:  # X, coords
                    X_batch, coords_batch = batch
                    X_batch = X_batch.to(self.device)
                    coords_batch = coords_batch.to(self.device)
                    batch_predictions = self.model(X_batch, coords_batch)
                else:  # X only
                    X_batch = batch[0].to(self.device)
                    batch_predictions = self.model(X_batch)
                
                predictions.append(batch_predictions.cpu())
        
        return torch.cat(predictions, dim=0)
    
    def evaluate(self,
                X: torch.Tensor,
                y: torch.Tensor,
                coords: Optional[torch.Tensor] = None) -> ModelMetrics:
        """
        Оценивает модель на тестовых данных.
        
        Args:
            X: Признаки [N, features]
            y: Истинные значения [N, 1] или [N]
            coords: Координаты атомов [N, atoms, 3] (для EGNN)
        
        Returns:
            ModelMetrics: Метрики модели
        """
        predictions = self.predict(X, coords)
        
        if len(y.shape) == 1:
            y = y.unsqueeze(1)
        
        metrics = self.metrics_calculator.calculate_metrics(y, predictions)
        
        logger.info(f"Оценка модели: {metrics}")
        
        return metrics
    
    def load_best_model(self):
        """Загружает лучшую сохраненную модель."""
        best_model_path = self.save_dir / "best_model.pth"
        if best_model_path.exists():
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Загружена лучшая модель с эпохи {checkpoint['epoch']}")
        else:
            logger.warning("Лучшая модель не найдена")
    
    def save_model(self, path: str):
        """
        Сохраняет текущую модель.
        
        Args:
            path: Путь для сохранения
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'experiment_name': self.experiment_name
        }, path)
        
        logger.info(f"Модель сохранена: {path}")
    
    def load_model(self, path: str):
        """
        Загружает модель из файла.
        
        Args:
            path: Путь к файлу модели
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Модель загружена: {path}")