"""
Метрики для оценки качества моделей молекулярного машинного обучения.

Содержит реализации MAE, RMSE, R² и других метрик,
специально адаптированных для химических свойств.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Структура для хранения метрик модели."""
    
    # Основные метрики регрессии
    mae: float
    rmse: float
    r2: float
    
    # Дополнительные метрики
    mse: float
    max_error: float
    mean_error: float
    std_error: float
    
    # Метаданные
    n_samples: int
    target_mean: float
    target_std: float
    
    def __str__(self) -> str:
        """Строковое представление метрик."""
        return (f"ModelMetrics(MAE={self.mae:.4f}, RMSE={self.rmse:.4f}, "
                f"R²={self.r2:.4f}, n_samples={self.n_samples})")
    
    def to_dict(self) -> Dict[str, float]:
        """Конвертирует метрики в словарь."""
        return {
            'mae': self.mae,
            'rmse': self.rmse,
            'r2': self.r2,
            'mse': self.mse,
            'max_error': self.max_error,
            'mean_error': self.mean_error,
            'std_error': self.std_error,
            'n_samples': self.n_samples,
            'target_mean': self.target_mean,
            'target_std': self.target_std
        }


class MetricsCalculator:
    """
    Калькулятор метрик для молекулярного машинного обучения.
    
    Вычисляет стандартные метрики регрессии (MAE, RMSE, R²) и
    дополнительные метрики, важные для химических свойств.
    """
    
    def __init__(self, 
                 property_name: str = "molecular_property",
                 property_units: str = "eV"):
        """
        Инициализация калькулятора метрик.
        
        Args:
            property_name: Название предсказываемого свойства
            property_units: Единицы измерения свойства
        """
        self.property_name = property_name
        self.property_units = property_units
        
        logger.info(f"Инициализирован MetricsCalculator для {property_name} ({property_units})")
    
    def calculate_metrics(self, 
                         y_true: torch.Tensor,
                         y_pred: torch.Tensor) -> ModelMetrics:
        """
        Вычисляет все метрики для предсказаний.
        
        Args:
            y_true: Истинные значения [N, 1] или [N]
            y_pred: Предсказанные значения [N, 1] или [N]
        
        Returns:
            ModelMetrics: Структура с вычисленными метриками
        """
        # Конвертируем в numpy и приводим к одномерному виду
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        
        # Проверяем размерности
        if len(y_true) != len(y_pred):
            raise ValueError(f"Размерности не совпадают: y_true={len(y_true)}, y_pred={len(y_pred)}")
        
        if len(y_true) == 0:
            raise ValueError("Пустые массивы предсказаний")
        
        # Вычисляем ошибки
        errors = y_pred - y_true
        
        # Основные метрики
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        
        # R² с обработкой особых случаев
        try:
            r2 = r2_score(y_true, y_pred)
            # Проверяем на NaN или inf
            if not np.isfinite(r2):
                logger.warning("R² не является конечным числом, устанавливаем в -inf")
                r2 = float('-inf')
        except Exception as e:
            logger.warning(f"Ошибка при вычислении R²: {e}, устанавливаем в -inf")
            r2 = float('-inf')
        
        # Дополнительные метрики
        max_error = np.max(np.abs(errors))
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        # Статистики целевой переменной
        target_mean = np.mean(y_true)
        target_std = np.std(y_true)
        
        metrics = ModelMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2,
            mse=mse,
            max_error=max_error,
            mean_error=mean_error,
            std_error=std_error,
            n_samples=len(y_true),
            target_mean=target_mean,
            target_std=target_std
        )
        
        logger.debug(f"Вычислены метрики: {metrics}")
        
        return metrics
    
    def calculate_relative_metrics(self, 
                                  y_true: torch.Tensor,
                                  y_pred: torch.Tensor) -> Dict[str, float]:
        """
        Вычисляет относительные метрики (в процентах от среднего значения).
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
        
        Returns:
            Dict[str, float]: Относительные метрики
        """
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Избегаем деления на ноль
        if abs(metrics.target_mean) < 1e-8:
            logger.warning("Среднее значение близко к нулю, относительные метрики могут быть неточными")
            mean_abs = np.mean(np.abs(y_true.detach().cpu().numpy().flatten()))
            if mean_abs < 1e-8:
                mean_abs = 1.0  # Fallback значение
        else:
            mean_abs = abs(metrics.target_mean)
        
        relative_metrics = {
            'relative_mae_percent': (metrics.mae / mean_abs) * 100,
            'relative_rmse_percent': (metrics.rmse / mean_abs) * 100,
            'relative_max_error_percent': (metrics.max_error / mean_abs) * 100
        }
        
        return relative_metrics
    
    def compare_models(self, 
                      models_metrics: Dict[str, ModelMetrics]) -> Dict[str, Any]:
        """
        Сравнивает метрики нескольких моделей.
        
        Args:
            models_metrics: Словарь {model_name: ModelMetrics}
        
        Returns:
            Dict[str, Any]: Результаты сравнения
        """
        if not models_metrics:
            raise ValueError("Нет метрик для сравнения")
        
        # Находим лучшую модель по каждой метрике
        best_models = {}
        
        # MAE (меньше = лучше)
        best_mae_model = min(models_metrics.keys(), key=lambda k: models_metrics[k].mae)
        best_models['mae'] = {
            'model': best_mae_model,
            'value': models_metrics[best_mae_model].mae
        }
        
        # RMSE (меньше = лучше)
        best_rmse_model = min(models_metrics.keys(), key=lambda k: models_metrics[k].rmse)
        best_models['rmse'] = {
            'model': best_rmse_model,
            'value': models_metrics[best_rmse_model].rmse
        }
        
        # R² (больше = лучше)
        valid_r2_models = {k: v for k, v in models_metrics.items() if np.isfinite(v.r2)}
        if valid_r2_models:
            best_r2_model = max(valid_r2_models.keys(), key=lambda k: valid_r2_models[k].r2)
            best_models['r2'] = {
                'model': best_r2_model,
                'value': valid_r2_models[best_r2_model].r2
            }
        else:
            best_models['r2'] = {'model': 'None', 'value': float('-inf')}
        
        # Создаем сводную таблицу
        comparison_table = []
        for model_name, metrics in models_metrics.items():
            comparison_table.append({
                'model': model_name,
                'mae': metrics.mae,
                'rmse': metrics.rmse,
                'r2': metrics.r2,
                'n_samples': metrics.n_samples
            })
        
        # Сортируем по MAE (основная метрика)
        comparison_table.sort(key=lambda x: x['mae'])
        
        return {
            'best_models': best_models,
            'comparison_table': comparison_table,
            'summary': self._create_comparison_summary(models_metrics, best_models)
        }
    
    def _create_comparison_summary(self, 
                                  models_metrics: Dict[str, ModelMetrics],
                                  best_models: Dict[str, Dict[str, Any]]) -> str:
        """Создает текстовое резюме сравнения моделей."""
        summary = f"\n=== Сравнение моделей для {self.property_name} ===\n"
        
        summary += f"Количество моделей: {len(models_metrics)}\n"
        summary += f"Размер тестового набора: {list(models_metrics.values())[0].n_samples}\n\n"
        
        summary += "Лучшие модели по метрикам:\n"
        summary += f"• MAE: {best_models['mae']['model']} ({best_models['mae']['value']:.4f} {self.property_units})\n"
        summary += f"• RMSE: {best_models['rmse']['model']} ({best_models['rmse']['value']:.4f} {self.property_units})\n"
        summary += f"• R²: {best_models['r2']['model']} ({best_models['r2']['value']:.4f})\n\n"
        
        # Добавляем интерпретацию R²
        best_r2 = best_models['r2']['value']
        if best_r2 >= 0.9:
            r2_interpretation = "отличное качество"
        elif best_r2 >= 0.7:
            r2_interpretation = "хорошее качество"
        elif best_r2 >= 0.5:
            r2_interpretation = "удовлетворительное качество"
        elif best_r2 >= 0.0:
            r2_interpretation = "плохое качество"
        else:
            r2_interpretation = "очень плохое качество (хуже константного предсказания)"
        
        summary += f"Интерпретация лучшего R²: {r2_interpretation}\n"
        
        return summary
    
    def plot_predictions(self, 
                        y_true: torch.Tensor,
                        y_pred: torch.Tensor,
                        model_name: str = "Model",
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Создает график предсказаний vs истинных значений.
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            model_name: Название модели
            save_path: Путь для сохранения графика
        
        Returns:
            plt.Figure: График
        """
        # Конвертируем в numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().cpu().numpy().flatten()
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy().flatten()
        
        # Вычисляем метрики
        metrics = self.calculate_metrics(torch.tensor(y_true), torch.tensor(y_pred))
        
        # Создаем график
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # График 1: Предсказания vs истинные значения
        ax1.scatter(y_true, y_pred, alpha=0.6, s=20)
        
        # Линия идеального предсказания
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Идеальное предсказание')
        
        ax1.set_xlabel(f'Истинные значения ({self.property_units})')
        ax1.set_ylabel(f'Предсказанные значения ({self.property_units})')
        ax1.set_title(f'{model_name}: Предсказания vs Истинные значения')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Добавляем метрики на график
        textstr = f'MAE = {metrics.mae:.4f}\nRMSE = {metrics.rmse:.4f}\nR² = {metrics.r2:.4f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # График 2: Распределение ошибок
        errors = y_pred - y_true
        ax2.hist(errors, bins=30, alpha=0.7, edgecolor='black')
        ax2.axvline(0, color='red', linestyle='--', alpha=0.8, label='Нулевая ошибка')
        ax2.axvline(np.mean(errors), color='green', linestyle='-', alpha=0.8, 
                   label=f'Средняя ошибка = {np.mean(errors):.4f}')
        
        ax2.set_xlabel(f'Ошибка предсказания ({self.property_units})')
        ax2.set_ylabel('Частота')
        ax2.set_title(f'{model_name}: Распределение ошибок')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"График сохранен: {save_path}")
        
        return fig
    
    def plot_model_comparison(self, 
                             models_metrics: Dict[str, ModelMetrics],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Создает график сравнения моделей.
        
        Args:
            models_metrics: Словарь {model_name: ModelMetrics}
            save_path: Путь для сохранения графика
        
        Returns:
            plt.Figure: График сравнения
        """
        if not models_metrics:
            raise ValueError("Нет метрик для визуализации")
        
        model_names = list(models_metrics.keys())
        mae_values = [metrics.mae for metrics in models_metrics.values()]
        rmse_values = [metrics.rmse for metrics in models_metrics.values()]
        r2_values = [metrics.r2 if np.isfinite(metrics.r2) else 0 for metrics in models_metrics.values()]
        
        # Создаем график
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # MAE
        bars1 = axes[0].bar(model_names, mae_values, alpha=0.7, color='skyblue')
        axes[0].set_ylabel(f'MAE ({self.property_units})')
        axes[0].set_title('Mean Absolute Error')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3)
        
        # Добавляем значения на столбцы
        for bar, value in zip(bars1, mae_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(mae_values)*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # RMSE
        bars2 = axes[1].bar(model_names, rmse_values, alpha=0.7, color='lightcoral')
        axes[1].set_ylabel(f'RMSE ({self.property_units})')
        axes[1].set_title('Root Mean Square Error')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, rmse_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_values)*0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # R²
        bars3 = axes[2].bar(model_names, r2_values, alpha=0.7, color='lightgreen')
        axes[2].set_ylabel('R²')
        axes[2].set_title('Coefficient of Determination')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.5, label='Baseline (константное предсказание)')
        axes[2].legend()
        
        for bar, value in zip(bars3, r2_values):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(r2_values) - min(r2_values))*0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.suptitle(f'Сравнение моделей: {self.property_name}', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"График сравнения сохранен: {save_path}")
        
        return fig
    
    def interpret_metrics(self, metrics: ModelMetrics) -> str:
        """
        Создает интерпретацию метрик для химических свойств.
        
        Args:
            metrics: Метрики модели
        
        Returns:
            str: Текстовая интерпретация
        """
        interpretation = f"\n=== Интерпретация метрик для {self.property_name} ===\n"
        
        # MAE интерпретация
        interpretation += f"MAE = {metrics.mae:.4f} {self.property_units}\n"
        interpretation += f"  → В среднем модель ошибается на {metrics.mae:.4f} {self.property_units}\n"
        
        # Относительная ошибка
        if abs(metrics.target_mean) > 1e-8:
            relative_mae = (metrics.mae / abs(metrics.target_mean)) * 100
            interpretation += f"  → Это составляет {relative_mae:.1f}% от среднего значения\n"
        
        # RMSE интерпретация
        interpretation += f"\nRMSE = {metrics.rmse:.4f} {self.property_units}\n"
        interpretation += f"  → Типичная ошибка с учетом выбросов\n"
        
        # Сравнение MAE и RMSE
        rmse_mae_ratio = metrics.rmse / metrics.mae if metrics.mae > 0 else float('inf')
        if rmse_mae_ratio > 1.5:
            interpretation += f"  → RMSE/MAE = {rmse_mae_ratio:.2f} - есть значительные выбросы\n"
        elif rmse_mae_ratio > 1.2:
            interpretation += f"  → RMSE/MAE = {rmse_mae_ratio:.2f} - умеренные выбросы\n"
        else:
            interpretation += f"  → RMSE/MAE = {rmse_mae_ratio:.2f} - ошибки распределены равномерно\n"
        
        # R² интерпретация
        interpretation += f"\nR² = {metrics.r2:.4f}\n"
        if metrics.r2 >= 0.9:
            interpretation += "  → Отличное качество: модель объясняет >90% дисперсии\n"
        elif metrics.r2 >= 0.7:
            interpretation += "  → Хорошее качество: модель объясняет >70% дисперсии\n"
        elif metrics.r2 >= 0.5:
            interpretation += "  → Удовлетворительное качество: модель объясняет >50% дисперсии\n"
        elif metrics.r2 >= 0.0:
            interpretation += "  → Плохое качество: модель лучше константного предсказания\n"
        else:
            interpretation += "  → Очень плохое качество: модель хуже константного предсказания\n"
        
        # Дополнительная информация
        interpretation += f"\nДополнительная информация:\n"
        interpretation += f"  • Размер выборки: {metrics.n_samples}\n"
        interpretation += f"  • Среднее значение цели: {metrics.target_mean:.4f} {self.property_units}\n"
        interpretation += f"  • Стандартное отклонение цели: {metrics.target_std:.4f} {self.property_units}\n"
        interpretation += f"  • Максимальная ошибка: {metrics.max_error:.4f} {self.property_units}\n"
        
        return interpretation