"""
Модуль для обучения и оценки моделей молекулярного машинного обучения.

Содержит классы для обучения моделей, вычисления метрик,
и управления процессом обучения.
"""

from .metrics import MetricsCalculator, ModelMetrics
from .trainer import ModelTrainer, TrainingConfig, TrainingHistory
from .utils import (
    EarlyStopping, 
    LearningRateScheduler, 
    CheckpointManager,
    TrainingLogger,
    GradientClipper,
    count_parameters,
    set_random_seed,
    get_device
)

__all__ = [
    # Метрики
    'MetricsCalculator',
    'ModelMetrics',
    
    # Обучение
    'ModelTrainer',
    'TrainingConfig', 
    'TrainingHistory',
    
    # Утилиты
    'EarlyStopping',
    'LearningRateScheduler',
    'CheckpointManager',
    'TrainingLogger',
    'GradientClipper',
    'count_parameters',
    'set_random_seed',
    'get_device'
]