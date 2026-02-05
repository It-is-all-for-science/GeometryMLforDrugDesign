"""
Специализированные метрики для оценки предсказания аффинности связывания.

Содержит метрики, специфичные для задач drug design и белок-лигандного связывания,
включая корреляции с экспериментальными данными и оценки практической применимости.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BindingAffinityMetrics:
    """Результаты оценки предсказания аффинности связывания."""
    
    # Основные метрики регрессии
    mae: float                    # Mean Absolute Error
    rmse: float                   # Root Mean Square Error
    r2: float                     # R-squared
    
    # Корреляционные метрики
    pearson_r: float              # Корреляция Пирсона
    pearson_p: float              # P-value для корреляции Пирсона
    spearman_r: float             # Корреляция Спирмена
    spearman_p: float             # P-value для корреляции Спирмена
    
    # Метрики для drug design
    top_1_percent_recall: float   # Recall в топ-1% предсказаний
    top_5_percent_recall: float   # Recall в топ-5% предсказаний
    enrichment_factor_1: float    # Фактор обогащения для топ-1%
    enrichment_factor_5: float    # Фактор обогащения для топ-5%
    
    # Практические метрики
    strong_binders_precision: float  # Точность для сильных связывателей (pKd > 8)
    strong_binders_recall: float     # Полнота для сильных связывателей
    weak_binders_precision: float    # Точность для слабых связывателей (pKd < 6)
    
    # Статистические метрики
    mean_error: float             # Средняя ошибка (bias)
    std_error: float              # Стандартное отклонение ошибки
    
    def __str__(self) -> str:
        """Строковое представление метрик."""
        return (f"BindingAffinityMetrics(\n"
                f"  MAE: {self.mae:.3f}, RMSE: {self.rmse:.3f}, R²: {self.r2:.3f}\n"
                f"  Pearson r: {self.pearson_r:.3f} (p={self.pearson_p:.3e})\n"
                f"  Spearman r: {self.spearman_r:.3f} (p={self.spearman_p:.3e})\n"
                f"  Top-1% Recall: {self.top_1_percent_recall:.3f}\n"
                f"  Strong Binders Precision: {self.strong_binders_precision:.3f}\n"
                f")")


class BindingAffinityEvaluator:
    """
    Класс для оценки качества предсказания аффинности связывания.
    
    Предоставляет специализированные метрики для задач drug design,
    учитывающие специфику белок-лигандного связывания.
    """
    
    def __init__(self, 
                 strong_binder_threshold: float = 8.0,
                 weak_binder_threshold: float = 6.0):
        """
        Инициализация оценщика.
        
        Args:
            strong_binder_threshold: Порог для сильных связывателей (pKd)
            weak_binder_threshold: Порог для слабых связывателей (pKd)
        """
        self.strong_binder_threshold = strong_binder_threshold
        self.weak_binder_threshold = weak_binder_threshold
        
        logger.info(f"Инициализирован BindingAffinityEvaluator "
                   f"(strong_threshold={strong_binder_threshold}, "
                   f"weak_threshold={weak_binder_threshold})")
    
    def evaluate(self, 
                y_true: np.ndarray, 
                y_pred: np.ndarray) -> BindingAffinityMetrics:
        """
        Вычисляет все метрики для предсказания аффинности.
        
        Args:
            y_true: Истинные значения аффинности (pKd)
            y_pred: Предсказанные значения аффинности (pKd)
        
        Returns:
            BindingAffinityMetrics: Полный набор метрик
        """
        if len(y_true) != len(y_pred):
            raise ValueError("Длины y_true и y_pred должны совпадать")
        
        if len(y_true) == 0:
            raise ValueError("Пустые массивы данных")
        
        # Основные метрики регрессии
        mae = self._calculate_mae(y_true, y_pred)
        rmse = self._calculate_rmse(y_true, y_pred)
        r2 = self._calculate_r2(y_true, y_pred)
        
        # Корреляционные метрики
        pearson_r, pearson_p = self._calculate_pearson_correlation(y_true, y_pred)
        spearman_r, spearman_p = self._calculate_spearman_correlation(y_true, y_pred)
        
        # Метрики для drug design
        top_1_recall = self._calculate_top_k_recall(y_true, y_pred, k=0.01)
        top_5_recall = self._calculate_top_k_recall(y_true, y_pred, k=0.05)
        enrichment_1 = self._calculate_enrichment_factor(y_true, y_pred, k=0.01)
        enrichment_5 = self._calculate_enrichment_factor(y_true, y_pred, k=0.05)
        
        # Практические метрики
        strong_precision, strong_recall = self._calculate_binder_metrics(
            y_true, y_pred, self.strong_binder_threshold, mode='strong'
        )
        weak_precision, _ = self._calculate_binder_metrics(
            y_true, y_pred, self.weak_binder_threshold, mode='weak'
        )
        
        # Статистические метрики
        errors = y_pred - y_true
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        return BindingAffinityMetrics(
            mae=mae,
            rmse=rmse,
            r2=r2,
            pearson_r=pearson_r,
            pearson_p=pearson_p,
            spearman_r=spearman_r,
            spearman_p=spearman_p,
            top_1_percent_recall=top_1_recall,
            top_5_percent_recall=top_5_recall,
            enrichment_factor_1=enrichment_1,
            enrichment_factor_5=enrichment_5,
            strong_binders_precision=strong_precision,
            strong_binders_recall=strong_recall,
            weak_binders_precision=weak_precision,
            mean_error=mean_error,
            std_error=std_error
        )
    
    def _calculate_mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Вычисляет Mean Absolute Error."""
        return np.mean(np.abs(y_true - y_pred))
    
    def _calculate_rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Вычисляет Root Mean Square Error."""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def _calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Вычисляет коэффициент детерминации R²."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        
        if ss_tot == 0:
            return 0.0
        
        return 1 - (ss_res / ss_tot)
    
    def _calculate_pearson_correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Вычисляет корреляцию Пирсона."""
        try:
            r, p = pearsonr(y_true, y_pred)
            return float(r), float(p)
        except:
            return 0.0, 1.0
    
    def _calculate_spearman_correlation(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
        """Вычисляет корреляцию Спирмена."""
        try:
            r, p = spearmanr(y_true, y_pred)
            return float(r), float(p)
        except:
            return 0.0, 1.0
    
    def _calculate_top_k_recall(self, y_true: np.ndarray, y_pred: np.ndarray, k: float) -> float:
        """
        Вычисляет recall в топ-k% предсказаний.
        
        Показывает, какую долю реально лучших соединений мы находим
        среди топ-k% предсказанных.
        """
        n = len(y_true)
        top_k_size = max(1, int(n * k))
        
        # Индексы топ-k% по истинным значениям
        true_top_indices = set(np.argsort(y_true)[-top_k_size:])
        
        # Индексы топ-k% по предсказанным значениям
        pred_top_indices = set(np.argsort(y_pred)[-top_k_size:])
        
        # Пересечение
        intersection = len(true_top_indices.intersection(pred_top_indices))
        
        return intersection / top_k_size
    
    def _calculate_enrichment_factor(self, y_true: np.ndarray, y_pred: np.ndarray, k: float) -> float:
        """
        Вычисляет фактор обогащения для топ-k%.
        
        Показывает, во сколько раз лучше случайного выбора наш метод
        находит хорошие соединения.
        """
        recall = self._calculate_top_k_recall(y_true, y_pred, k)
        return recall / k if k > 0 else 0.0
    
    def _calculate_binder_metrics(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray, 
                                threshold: float,
                                mode: str = 'strong') -> Tuple[float, float]:
        """
        Вычисляет precision и recall для связывателей определенной силы.
        
        Args:
            y_true: Истинные значения
            y_pred: Предсказанные значения
            threshold: Порог для классификации
            mode: 'strong' для сильных связывателей (>=threshold), 
                  'weak' для слабых (<threshold)
        
        Returns:
            Tuple[float, float]: precision, recall
        """
        if mode == 'strong':
            true_positives = (y_true >= threshold)
            pred_positives = (y_pred >= threshold)
        else:  # weak
            true_positives = (y_true < threshold)
            pred_positives = (y_pred < threshold)
        
        # True positives: правильно предсказанные положительные
        tp = np.sum(true_positives & pred_positives)
        
        # False positives: неправильно предсказанные как положительные
        fp = np.sum(~true_positives & pred_positives)
        
        # False negatives: пропущенные положительные
        fn = np.sum(true_positives & ~pred_positives)
        
        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        return precision, recall
    
    def create_performance_report(self, 
                                metrics: BindingAffinityMetrics,
                                model_name: str = "Model") -> str:
        """
        Создает текстовый отчет о производительности модели.
        
        Args:
            metrics: Метрики модели
            model_name: Название модели
        
        Returns:
            str: Форматированный отчет
        """
        report = f"""
# Отчет о производительности модели: {model_name}

## Основные метрики регрессии
- **MAE**: {metrics.mae:.3f} (средняя абсолютная ошибка в pKd единицах)
- **RMSE**: {metrics.rmse:.3f} (корень из средней квадратичной ошибки)
- **R²**: {metrics.r2:.3f} (коэффициент детерминации)

## Корреляционный анализ
- **Корреляция Пирсона**: r = {metrics.pearson_r:.3f} (p = {metrics.pearson_p:.3e})
- **Корреляция Спирмена**: r = {metrics.spearman_r:.3f} (p = {metrics.spearman_p:.3e})

## Метрики для drug design
- **Top-1% Recall**: {metrics.top_1_percent_recall:.3f} (доля лучших соединений в топ-1% предсказаний)
- **Top-5% Recall**: {metrics.top_5_percent_recall:.3f} (доля лучших соединений в топ-5% предсказаний)
- **Enrichment Factor (1%)**: {metrics.enrichment_factor_1:.2f}x (улучшение по сравнению со случайным выбором)
- **Enrichment Factor (5%)**: {metrics.enrichment_factor_5:.2f}x

## Практические метрики
- **Precision для сильных связывателей**: {metrics.strong_binders_precision:.3f}
- **Recall для сильных связывателей**: {metrics.strong_binders_recall:.3f}
- **Precision для слабых связывателей**: {metrics.weak_binders_precision:.3f}

## Статистический анализ ошибок
- **Средняя ошибка (bias)**: {metrics.mean_error:.3f} pKd
- **Стандартное отклонение ошибки**: {metrics.std_error:.3f} pKd

## Интерпретация результатов

### Качество регрессии
"""
        
        # Интерпретация R²
        if metrics.r2 > 0.8:
            report += "- ✅ **Отличное качество регрессии** (R² > 0.8)\n"
        elif metrics.r2 > 0.6:
            report += "- ✅ **Хорошее качество регрессии** (R² > 0.6)\n"
        elif metrics.r2 > 0.4:
            report += "- ⚠️ **Удовлетворительное качество регрессии** (R² > 0.4)\n"
        else:
            report += "- ❌ **Низкое качество регрессии** (R² < 0.4)\n"
        
        # Интерпретация MAE
        if metrics.mae < 0.5:
            report += "- ✅ **Низкая средняя ошибка** (MAE < 0.5 pKd)\n"
        elif metrics.mae < 1.0:
            report += "- ✅ **Приемлемая средняя ошибка** (MAE < 1.0 pKd)\n"
        else:
            report += "- ⚠️ **Высокая средняя ошибка** (MAE > 1.0 pKd)\n"
        
        report += "\n### Применимость для drug design\n"
        
        # Интерпретация enrichment factor
        if metrics.enrichment_factor_1 > 5:
            report += "- ✅ **Отличное обогащение** - модель эффективно находит лучшие соединения\n"
        elif metrics.enrichment_factor_1 > 2:
            report += "- ✅ **Хорошее обогащение** - модель полезна для виртуального скрининга\n"
        else:
            report += "- ⚠️ **Слабое обогащение** - модель мало лучше случайного выбора\n"
        
        # Интерпретация precision для сильных связывателей
        if metrics.strong_binders_precision > 0.7:
            report += "- ✅ **Высокая точность** для сильных связывателей\n"
        elif metrics.strong_binders_precision > 0.5:
            report += "- ✅ **Приемлемая точность** для сильных связывателей\n"
        else:
            report += "- ⚠️ **Низкая точность** для сильных связывателей\n"
        
        return report


def compare_binding_models(models_metrics: Dict[str, BindingAffinityMetrics]) -> str:
    """
    Сравнивает несколько моделей по метрикам аффинности связывания.
    
    Args:
        models_metrics: Словарь {название_модели: метрики}
    
    Returns:
        str: Сравнительный отчет
    """
    if not models_metrics:
        return "Нет данных для сравнения"
    
    report = "# Сравнительный анализ моделей предсказания аффинности\n\n"
    
    # Создаем таблицу сравнения
    report += "## Сводная таблица метрик\n\n"
    report += "| Модель | MAE | RMSE | R² | Pearson r | Top-1% Recall | EF(1%) |\n"
    report += "|--------|-----|------|----|-----------|--------------:|--------:|\n"
    
    for model_name, metrics in models_metrics.items():
        report += (f"| {model_name} | {metrics.mae:.3f} | {metrics.rmse:.3f} | "
                  f"{metrics.r2:.3f} | {metrics.pearson_r:.3f} | "
                  f"{metrics.top_1_percent_recall:.3f} | {metrics.enrichment_factor_1:.1f}x |\n")
    
    # Определяем лучшие модели по разным критериям
    report += "\n## Лучшие модели по критериям\n\n"
    
    best_r2 = max(models_metrics.items(), key=lambda x: x[1].r2)
    best_mae = min(models_metrics.items(), key=lambda x: x[1].mae)
    best_enrichment = max(models_metrics.items(), key=lambda x: x[1].enrichment_factor_1)
    best_precision = max(models_metrics.items(), key=lambda x: x[1].strong_binders_precision)
    
    report += f"- **Лучшая по R²**: {best_r2[0]} (R² = {best_r2[1].r2:.3f})\n"
    report += f"- **Лучшая по MAE**: {best_mae[0]} (MAE = {best_mae[1].mae:.3f})\n"
    report += f"- **Лучшая по обогащению**: {best_enrichment[0]} (EF = {best_enrichment[1].enrichment_factor_1:.1f}x)\n"
    report += f"- **Лучшая по точности**: {best_precision[0]} (Precision = {best_precision[1].strong_binders_precision:.3f})\n"
    
    return report


# Пример использования
if __name__ == "__main__":
    # Создаем синтетические данные для тестирования
    np.random.seed(42)
    n_samples = 1000
    
    # Истинные значения аффинности (pKd от 4 до 12)
    y_true = np.random.uniform(4, 12, n_samples)
    
    # Предсказания с некоторым шумом
    y_pred = y_true + np.random.normal(0, 0.5, n_samples)
    
    # Создаем оценщик
    evaluator = BindingAffinityEvaluator()
    
    # Вычисляем метрики
    metrics = evaluator.evaluate(y_true, y_pred)
    
    # Выводим результаты
    print(metrics)
    
    # Создаем отчет
    report = evaluator.create_performance_report(metrics, "Test Model")
    print(report)