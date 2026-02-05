"""
Вспомогательные утилиты для обучения моделей.

Содержит классы для early stopping, learning rate scheduling,
управления чекпоинтами и другие полезные инструменты.
"""

import torch
import torch.optim as optim
import numpy as np
from typing import Optional, Dict, Any, List
import logging
import os
from pathlib import Path
import json
import time

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Реализация early stopping для предотвращения переобучения.
    
    Останавливает обучение, если метрика не улучшается в течение
    заданного количества эпох.
    """
    
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 1e-4,
                 mode: str = 'min',
                 verbose: bool = True):
        """
        Инициализация early stopping.
        
        Args:
            patience: Количество эпох без улучшения до остановки
            min_delta: Минимальное изменение для считания улучшением
            mode: 'min' для loss, 'max' для accuracy
            verbose: Выводить ли сообщения
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        # Функция сравнения в зависимости от режима
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
        else:
            self.is_better = lambda current, best: current > best + min_delta
        
        logger.info(f"Инициализирован EarlyStopping: patience={patience}, min_delta={min_delta}, mode={mode}")
    
    def __call__(self, score: float) -> bool:
        """
        Проверяет, нужно ли остановить обучение.
        
        Args:
            score: Текущее значение метрики
        
        Returns:
            bool: True если нужно остановить обучение
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.is_better(score, self.best_score):
            # Улучшение найдено
            self.best_score = score
            self.counter = 0
            if self.verbose:
                logger.debug(f"EarlyStopping: улучшение найдено, сброс счетчика")
        else:
            # Улучшения нет
            self.counter += 1
            if self.verbose:
                logger.debug(f"EarlyStopping: нет улучшения {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    logger.info(f"EarlyStopping: остановка обучения после {self.counter} эпох без улучшения")
        
        return self.early_stop
    
    def should_stop(self, score: float) -> bool:
        """Альтернативный интерфейс для проверки остановки."""
        return self.__call__(score)
    
    def reset(self):
        """Сбрасывает состояние early stopping."""
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        logger.info("EarlyStopping сброшен")


class LearningRateScheduler:
    """
    Обертка для PyTorch learning rate schedulers с дополнительной функциональностью.
    """
    
    def __init__(self,
                 optimizer: optim.Optimizer,
                 mode: str = 'min',
                 factor: float = 0.5,
                 patience: int = 5,
                 threshold: float = 1e-4,
                 min_lr: float = 1e-8,
                 verbose: bool = True):
        """
        Инициализация scheduler'а.
        
        Args:
            optimizer: Оптимизатор PyTorch
            mode: 'min' для loss, 'max' для accuracy
            factor: Коэффициент уменьшения learning rate
            patience: Количество эпох без улучшения до уменьшения LR
            threshold: Минимальное изменение для считания улучшением
            min_lr: Минимальное значение learning rate
            verbose: Выводить ли сообщения
        """
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            min_lr=min_lr
        )
        
        self.optimizer = optimizer
        self.verbose = verbose
        self.history = []
        
        logger.info(f"Инициализирован LearningRateScheduler: factor={factor}, patience={patience}")
    
    def step(self, metric: float):
        """
        Обновляет learning rate на основе метрики.
        
        Args:
            metric: Значение метрики для мониторинга
        """
        old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(metric)
        new_lr = self.optimizer.param_groups[0]['lr']
        
        self.history.append({
            'metric': metric,
            'lr': new_lr
        })
        
        if old_lr != new_lr and self.verbose:
            logger.info(f"Learning rate изменен: {old_lr:.2e} -> {new_lr:.2e}")
    
    def get_last_lr(self) -> float:
        """Возвращает текущий learning rate."""
        return self.optimizer.param_groups[0]['lr']


class CheckpointManager:
    """
    Управление сохранением и загрузкой чекпоинтов модели.
    """
    
    def __init__(self,
                 save_dir: str,
                 save_best: bool = True,
                 save_checkpoints: bool = True,
                 max_checkpoints: int = 5):
        """
        Инициализация менеджера чекпоинтов.
        
        Args:
            save_dir: Директория для сохранения
            save_best: Сохранять ли лучшую модель
            save_checkpoints: Сохранять ли промежуточные чекпоинты
            max_checkpoints: Максимальное количество чекпоинтов
        """
        self.save_dir = Path(save_dir)
        self.save_best = save_best
        self.save_checkpoints = save_checkpoints
        self.max_checkpoints = max_checkpoints
        
        # Создаем директорию
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Отслеживание чекпоинтов
        self.checkpoint_files = []
        self.best_score = None
        
        logger.info(f"Инициализирован CheckpointManager: {save_dir}")
    
    def save_checkpoint(self,
                       model: torch.nn.Module,
                       optimizer: optim.Optimizer,
                       epoch: int,
                       score: float,
                       additional_info: Optional[Dict[str, Any]] = None):
        """
        Сохраняет чекпоинт модели.
        
        Args:
            model: Модель для сохранения
            optimizer: Оптимизатор
            epoch: Номер эпохи
            score: Значение метрики
            additional_info: Дополнительная информация
        """
        if not self.save_checkpoints:
            return
        
        checkpoint_path = self.save_dir / f"checkpoint_epoch_{epoch:03d}.pth"
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score,
            'timestamp': time.time()
        }
        
        if additional_info:
            checkpoint_data.update(additional_info)
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Добавляем в список
        self.checkpoint_files.append({
            'path': checkpoint_path,
            'epoch': epoch,
            'score': score
        })
        
        # Удаляем старые чекпоинты
        self._cleanup_checkpoints()
        
        logger.debug(f"Сохранен чекпоинт: {checkpoint_path}")
    
    def save_best_model(self,
                       model: torch.nn.Module,
                       optimizer: optim.Optimizer,
                       epoch: int,
                       score: float,
                       additional_info: Optional[Dict[str, Any]] = None):
        """
        Сохраняет лучшую модель.
        
        Args:
            model: Модель для сохранения
            optimizer: Оптимизатор
            epoch: Номер эпохи
            score: Значение метрики
            additional_info: Дополнительная информация
        """
        if not self.save_best:
            return
        
        # Проверяем, является ли это лучшим результатом
        if self.best_score is None or score < self.best_score:
            self.best_score = score
            
            best_model_path = self.save_dir / "best_model.pth"
            
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'score': score,
                'timestamp': time.time()
            }
            
            if additional_info:
                checkpoint_data.update(additional_info)
            
            torch.save(checkpoint_data, best_model_path)
            
            logger.info(f"Сохранена лучшая модель: epoch={epoch}, score={score:.6f}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Загружает чекпоинт.
        
        Args:
            checkpoint_path: Путь к чекпоинту
        
        Returns:
            Dict[str, Any]: Данные чекпоинта
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        logger.info(f"Загружен чекпоинт: {checkpoint_path}")
        return checkpoint
    
    def load_best_model(self) -> Optional[Dict[str, Any]]:
        """
        Загружает лучшую модель.
        
        Returns:
            Optional[Dict[str, Any]]: Данные лучшей модели или None
        """
        best_model_path = self.save_dir / "best_model.pth"
        
        if best_model_path.exists():
            return self.load_checkpoint(str(best_model_path))
        else:
            logger.warning("Лучшая модель не найдена")
            return None
    
    def _cleanup_checkpoints(self):
        """Удаляет старые чекпоинты, оставляя только последние."""
        if len(self.checkpoint_files) > self.max_checkpoints:
            # Сортируем по эпохе
            self.checkpoint_files.sort(key=lambda x: x['epoch'])
            
            # Удаляем старые файлы
            files_to_remove = self.checkpoint_files[:-self.max_checkpoints]
            
            for file_info in files_to_remove:
                try:
                    file_info['path'].unlink()
                    logger.debug(f"Удален старый чекпоинт: {file_info['path']}")
                except FileNotFoundError:
                    pass
            
            # Обновляем список
            self.checkpoint_files = self.checkpoint_files[-self.max_checkpoints:]
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """
        Возвращает список доступных чекпоинтов.
        
        Returns:
            List[Dict[str, Any]]: Информация о чекпоинтах
        """
        return self.checkpoint_files.copy()


class TrainingLogger:
    """
    Логгер для отслеживания прогресса обучения.
    """
    
    def __init__(self,
                 log_dir: str,
                 experiment_name: str = "experiment"):
        """
        Инициализация логгера.
        
        Args:
            log_dir: Директория для логов
            experiment_name: Название эксперимента
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # Создаем директорию
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Файл для логов
        self.log_file = self.log_dir / f"{experiment_name}_training.log"
        
        # История метрик
        self.metrics_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': [],
            'learning_rates': [],
            'epochs': []
        }
        
        logger.info(f"Инициализирован TrainingLogger: {self.log_file}")
    
    def log_epoch(self,
                  epoch: int,
                  train_loss: float,
                  val_loss: Optional[float] = None,
                  train_metrics: Optional[Dict[str, float]] = None,
                  val_metrics: Optional[Dict[str, float]] = None,
                  learning_rate: Optional[float] = None):
        """
        Логирует результаты эпохи.
        
        Args:
            epoch: Номер эпохи
            train_loss: Loss на обучении
            val_loss: Loss на валидации
            train_metrics: Метрики на обучении
            val_metrics: Метрики на валидации
            learning_rate: Текущий learning rate
        """
        # Сохраняем в историю
        self.metrics_history['epochs'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['train_metrics'].append(train_metrics)
        self.metrics_history['val_metrics'].append(val_metrics)
        self.metrics_history['learning_rates'].append(learning_rate)
        
        # Записываем в файл
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'learning_rate': learning_rate,
            'timestamp': time.time()
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    def save_metrics_history(self):
        """Сохраняет историю метрик в JSON файл."""
        metrics_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
        
        logger.info(f"История метрик сохранена: {metrics_file}")
    
    def load_metrics_history(self) -> Dict[str, Any]:
        """
        Загружает историю метрик.
        
        Returns:
            Dict[str, Any]: История метрик
        """
        metrics_file = self.log_dir / f"{self.experiment_name}_metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                self.metrics_history = json.load(f)
            logger.info(f"История метрик загружена: {metrics_file}")
        else:
            logger.warning("Файл истории метрик не найден")
        
        return self.metrics_history


class GradientClipper:
    """
    Утилита для gradient clipping.
    """
    
    def __init__(self,
                 max_norm: float = 1.0,
                 norm_type: float = 2.0):
        """
        Инициализация gradient clipper.
        
        Args:
            max_norm: Максимальная норма градиентов
            norm_type: Тип нормы (1, 2, inf)
        """
        self.max_norm = max_norm
        self.norm_type = norm_type
        
        logger.info(f"Инициализирован GradientClipper: max_norm={max_norm}, norm_type={norm_type}")
    
    def clip_gradients(self, model: torch.nn.Module) -> float:
        """
        Обрезает градиенты модели.
        
        Args:
            model: Модель PyTorch
        
        Returns:
            float: Норма градиентов до обрезки
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type
        )
        
        return total_norm.item()


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Подсчитывает количество параметров в модели.
    
    Args:
        model: Модель PyTorch
    
    Returns:
        Dict[str, int]: Статистика параметров
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def set_random_seed(seed: int = 42):
    """
    Устанавливает случайное зерно для воспроизводимости.
    
    Args:
        seed: Значение зерна
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    # Для полной детерминированности (может замедлить обучение)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    logger.info(f"Установлено случайное зерно: {seed}")


def get_device(prefer_cuda: bool = True) -> torch.device:
    """
    Определяет оптимальное устройство для вычислений.
    
    Args:
        prefer_cuda: Предпочитать ли CUDA если доступна
    
    Returns:
        torch.device: Устройство для вычислений
    """
    if prefer_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Используется CUDA: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Используется CPU")
    
    return device