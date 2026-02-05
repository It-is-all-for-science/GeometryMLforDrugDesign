"""
Модуль для сравнения моделей и анализа результатов.

Содержит функции для:
- Сравнения производительности различных моделей
- Статистического анализа результатов
- Визуализации сравнений
- Анализа неопределенности
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import logging
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class ModelComparison:
    """Класс для сравнения моделей."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Инициализация компаратора моделей.
        
        Args:
            results_dir: Директория с результатами моделей
        """
        self.results_dir = Path(results_dir)
        self.model_results = {}
        self.comparison_data = {}
        
    def load_model_results(self, model_name: str, results_file: str) -> Dict:
        """
        Загружает результаты модели из файла.
        
        Args:
            model_name: Название модели
            results_file: Путь к файлу с результатами
            
        Returns:
            Словарь с результатами модели
        """
        results_path = self.results_dir / results_file
        
        if not results_path.exists():
            logger.warning(f"Файл результатов {results_path} не найден")
            return {}
            
        try:
            with open(results_path, 'r') as f:
                results = json.load(f)
                self.model_results[model_name] = results
                logger.info(f"Загружены результаты для модели {model_name}")
                return results
        except Exception as e:
            logger.error(f"Ошибка загрузки результатов {model_name}: {e}")
            return {}
    
    def compare_metrics(self, metric_name: str = 'mae') -> pd.DataFrame:
        """
        Сравнивает метрики между моделями.
        
        Args:
            metric_name: Название метрики для сравнения
            
        Returns:
            DataFrame с сравнением метрик
        """
        comparison_data = []
        
        for model_name, results in self.model_results.items():
            if 'test_metrics' in results:
                metrics = results['test_metrics']
                if metric_name in metrics:
                    comparison_data.append({
                        'model': model_name,
                        'metric': metric_name,
                        'value': metrics[metric_name]
                    })
        
        return pd.DataFrame(comparison_data)
    
    def statistical_comparison(self, predictions_dict: Dict[str, np.ndarray], 
                             true_values: np.ndarray) -> Dict:
        """
        Проводит статистическое сравнение предсказаний моделей.
        
        Args:
            predictions_dict: Словарь {model_name: predictions}
            true_values: Истинные значения
            
        Returns:
            Словарь со статистическими метриками
        """
        stats_results = {}
        
        for model_name, predictions in predictions_dict.items():
            mae = mean_absolute_error(true_values, predictions)
            mse = mean_squared_error(true_values, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(true_values, predictions)
            
            # Корреляция Пирсона
            pearson_r, pearson_p = stats.pearsonr(true_values, predictions)
            
            # Корреляция Спирмена
            spearman_r, spearman_p = stats.spearmanr(true_values, predictions)
            
            stats_results[model_name] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p
            }
        
        return stats_results
    
    def plot_model_comparison(self, metric_name: str = 'mae', 
                            save_path: Optional[str] = None) -> None:
        """
        Создает график сравнения моделей.
        
        Args:
            metric_name: Метрика для сравнения
            save_path: Путь для сохранения графика
        """
        df = self.compare_metrics(metric_name)
        
        if df.empty:
            logger.warning("Нет данных для построения графика")
            return
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df, x='model', y='value')
        plt.title(f'Сравнение моделей по метрике {metric_name.upper()}')
        plt.xlabel('Модель')
        plt.ylabel(f'{metric_name.upper()}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"График сохранен: {save_path}")
        
        plt.show()
    
    def create_comparison_report(self, output_file: str = "model_comparison_report.md") -> None:
        """
        Создает отчет сравнения моделей.
        
        Args:
            output_file: Имя файла для сохранения отчета
        """
        report_path = self.results_dir / output_file
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# Отчет сравнения моделей\n\n")
            
            # Общая информация
            f.write(f"Количество сравниваемых моделей: {len(self.model_results)}\n\n")
            
            # Таблица результатов
            f.write("## Сравнение метрик\n\n")
            
            for metric in ['mae', 'mse', 'r2']:
                df = self.compare_metrics(metric)
                if not df.empty:
                    f.write(f"### {metric.upper()}\n\n")
                    f.write(df.to_markdown(index=False))
                    f.write("\n\n")
            
            # Лучшая модель
            mae_df = self.compare_metrics('mae')
            if not mae_df.empty:
                best_model = mae_df.loc[mae_df['value'].idxmin()]
                f.write(f"## Лучшая модель\n\n")
                f.write(f"**{best_model['model']}** с MAE = {best_model['value']:.4f}\n\n")
        
        logger.info(f"Отчет сохранен: {report_path}")


def ensemble_predictions(predictions_dict: Dict[str, np.ndarray], 
                        method: str = 'mean') -> np.ndarray:
    """
    Создает ensemble предсказания из нескольких моделей.
    
    Args:
        predictions_dict: Словарь {model_name: predictions}
        method: Метод ensemble ('mean', 'median', 'weighted')
        
    Returns:
        Ensemble предсказания
    """
    predictions_array = np.array(list(predictions_dict.values()))
    
    if method == 'mean':
        return np.mean(predictions_array, axis=0)
    elif method == 'median':
        return np.median(predictions_array, axis=0)
    elif method == 'weighted':
        # Простое взвешивание - можно улучшить
        weights = np.ones(len(predictions_dict)) / len(predictions_dict)
        return np.average(predictions_array, axis=0, weights=weights)
    else:
        raise ValueError(f"Неизвестный метод ensemble: {method}")


def calculate_uncertainty(predictions_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Вычисляет неопределенность предсказаний ensemble.
    
    Args:
        predictions_dict: Словарь {model_name: predictions}
        
    Returns:
        Tuple (mean_predictions, uncertainty_std)
    """
    predictions_array = np.array(list(predictions_dict.values()))
    
    mean_predictions = np.mean(predictions_array, axis=0)
    uncertainty_std = np.std(predictions_array, axis=0)
    
    return mean_predictions, uncertainty_std