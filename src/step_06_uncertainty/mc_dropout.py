#!/usr/bin/env python3
"""
Monte Carlo Dropout для оценки epistemic uncertainty.

Epistemic uncertainty (неопределенность модели) возникает из-за недостатка данных
и может быть уменьшена с увеличением размера обучающей выборки.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


class MCDropoutPredictor:
    """
    Предсказатель с Monte Carlo Dropout для оценки uncertainty.
    
    Использует dropout во время inference для получения распределения предсказаний.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 n_samples: int = 100,
                 device: str = 'cuda'):
        """
        Инициализация MC Dropout предсказателя.
        
        Args:
            model: Обученная модель с dropout слоями
            n_samples: Количество forward passes для оценки uncertainty
            device: Устройство для вычислений
        """
        self.model = model
        self.n_samples = n_samples
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Включаем dropout во время inference
        self._enable_dropout()
        
        logger.info(f"Инициализирован MC Dropout с {n_samples} samples на {self.device}")
    
    def _enable_dropout(self):
        """Включает dropout слои в режиме eval."""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def predict_with_uncertainty(self,
                                 X: torch.Tensor,
                                 coords: Optional[torch.Tensor] = None,
                                 batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Делает предсказания с оценкой uncertainty.
        
        Args:
            X: Входные признаки
            coords: Координаты (для геометрических моделей)
            batch_size: Размер батча
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - mean_predictions: Средние предсказания
                - epistemic_uncertainty: Epistemic uncertainty (std)
                - all_predictions: Все предсказания (n_samples x n_data)
        """
        self.model.eval()
        self._enable_dropout()
        
        n_data = X.shape[0]
        all_predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                batch_predictions = []
                
                # Обрабатываем данные батчами
                for i in range(0, n_data, batch_size):
                    batch_X = X[i:i+batch_size].to(self.device)
                    
                    if coords is not None:
                        batch_coords = coords[i:i+batch_size].to(self.device)
                        predictions = self.model(batch_X, batch_coords)
                    else:
                        predictions = self.model(batch_X)
                    
                    batch_predictions.append(predictions.cpu().numpy())
                
                # Объединяем предсказания для всех батчей
                sample_predictions = np.concatenate(batch_predictions, axis=0)
                all_predictions.append(sample_predictions)
        
        # Преобразуем в numpy array: (n_samples, n_data, output_dim)
        all_predictions = np.array(all_predictions)
        
        # Вычисляем статистики
        mean_predictions = np.mean(all_predictions, axis=0)
        epistemic_uncertainty = np.std(all_predictions, axis=0)
        
        logger.info(f"MC Dropout: mean uncertainty = {np.mean(epistemic_uncertainty):.4f}")
        
        return mean_predictions, epistemic_uncertainty, all_predictions
    
    def predict_with_confidence_intervals(self,
                                         X: torch.Tensor,
                                         coords: Optional[torch.Tensor] = None,
                                         confidence: float = 0.95,
                                         batch_size: int = 32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Делает предсказания с доверительными интервалами.
        
        Args:
            X: Входные признаки
            coords: Координаты
            confidence: Уровень доверия (например, 0.95 для 95% CI)
            batch_size: Размер батча
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - mean_predictions: Средние предсказания
                - lower_bound: Нижняя граница доверительного интервала
                - upper_bound: Верхняя граница доверительного интервала
        """
        mean_pred, _, all_pred = self.predict_with_uncertainty(X, coords, batch_size)
        
        # Вычисляем перцентили для доверительных интервалов
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(all_pred, lower_percentile, axis=0)
        upper_bound = np.percentile(all_pred, upper_percentile, axis=0)
        
        return mean_pred, lower_bound, upper_bound


def analyze_uncertainty_distribution(epistemic_uncertainty: np.ndarray,
                                     predictions: np.ndarray,
                                     targets: np.ndarray) -> dict:
    """
    Анализирует распределение uncertainty и его связь с ошибками.
    
    Args:
        epistemic_uncertainty: Epistemic uncertainty для каждого примера
        predictions: Предсказания модели
        targets: Истинные значения
    
    Returns:
        dict: Статистики uncertainty
    """
    errors = np.abs(predictions.flatten() - targets.flatten())
    uncertainty = epistemic_uncertainty.flatten()
    
    # Корреляция между uncertainty и ошибкой
    correlation = np.corrcoef(uncertainty, errors)[0, 1]
    
    # Статистики
    stats = {
        'mean_uncertainty': float(np.mean(uncertainty)),
        'std_uncertainty': float(np.std(uncertainty)),
        'min_uncertainty': float(np.min(uncertainty)),
        'max_uncertainty': float(np.max(uncertainty)),
        'median_uncertainty': float(np.median(uncertainty)),
        'uncertainty_error_correlation': float(correlation),
        'high_uncertainty_threshold': float(np.percentile(uncertainty, 90)),
        'low_uncertainty_threshold': float(np.percentile(uncertainty, 10))
    }
    
    # Анализ по квантилям uncertainty
    n_quantiles = 5
    quantiles = np.linspace(0, 100, n_quantiles + 1)
    quantile_stats = []
    
    for i in range(n_quantiles):
        lower = np.percentile(uncertainty, quantiles[i])
        upper = np.percentile(uncertainty, quantiles[i + 1])
        
        mask = (uncertainty >= lower) & (uncertainty <= upper)
        quantile_errors = errors[mask]
        
        quantile_stats.append({
            'quantile': f'{quantiles[i]:.0f}-{quantiles[i+1]:.0f}%',
            'mean_error': float(np.mean(quantile_errors)),
            'std_error': float(np.std(quantile_errors)),
            'n_samples': int(np.sum(mask))
        })
    
    stats['quantile_analysis'] = quantile_stats
    
    logger.info(f"Uncertainty-Error correlation: {correlation:.3f}")
    
    return stats
