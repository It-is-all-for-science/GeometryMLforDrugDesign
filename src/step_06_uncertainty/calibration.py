#!/usr/bin/env python3
"""
Calibration analysis для проверки качества uncertainty estimates.

Хорошо откалиброванная модель должна давать uncertainty, которая соответствует
реальной частоте ошибок.
"""

import numpy as np
from typing import Tuple, List, Dict
import logging

logger = logging.getLogger(__name__)


def compute_calibration_curve(predictions: np.ndarray,
                              uncertainties: np.ndarray,
                              targets: np.ndarray,
                              n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Вычисляет calibration curve для regression задачи.
    
    Args:
        predictions: Предсказания модели
        uncertainties: Оценки uncertainty (std)
        targets: Истинные значения
        n_bins: Количество бинов для calibration curve
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - expected_frequencies: Ожидаемые частоты попадания в интервал
            - observed_frequencies: Наблюдаемые частоты
            - bin_centers: Центры бинов
    """
    predictions = predictions.flatten()
    uncertainties = uncertainties.flatten()
    targets = targets.flatten()
    
    # Вычисляем z-scores (нормализованные ошибки)
    errors = predictions - targets
    z_scores = np.abs(errors / (uncertainties + 1e-8))
    
    # Создаем бины для разных уровней confidence
    confidence_levels = np.linspace(0, 3, n_bins + 1)  # от 0 до 3 sigma
    expected_frequencies = []
    observed_frequencies = []
    bin_centers = []
    
    for i in range(n_bins):
        lower = confidence_levels[i]
        upper = confidence_levels[i + 1]
        
        # Ожидаемая частота (из нормального распределения)
        from scipy.stats import norm
        expected_freq = norm.cdf(upper) - norm.cdf(lower)
        
        # Наблюдаемая частота
        mask = (z_scores >= lower) & (z_scores < upper)
        observed_freq = np.mean(mask)
        
        expected_frequencies.append(expected_freq)
        observed_frequencies.append(observed_freq)
        bin_centers.append((lower + upper) / 2)
    
    return (np.array(expected_frequencies), 
            np.array(observed_frequencies),
            np.array(bin_centers))


def compute_calibration_metrics(predictions: np.ndarray,
                                uncertainties: np.ndarray,
                                targets: np.ndarray) -> Dict[str, float]:
    """
    Вычисляет метрики качества calibration.
    
    Args:
        predictions: Предсказания модели
        uncertainties: Оценки uncertainty
        targets: Истинные значения
    
    Returns:
        Dict[str, float]: Метрики calibration
    """
    predictions = predictions.flatten()
    uncertainties = uncertainties.flatten()
    targets = targets.flatten()
    
    errors = predictions - targets
    z_scores = errors / (uncertainties + 1e-8)
    
    # Expected Calibration Error (ECE)
    expected_freq, observed_freq, _ = compute_calibration_curve(
        predictions, uncertainties, targets, n_bins=10
    )
    ece = np.mean(np.abs(expected_freq - observed_freq))
    
    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(expected_freq - observed_freq))
    
    # Проверка нормальности z-scores
    from scipy.stats import shapiro, kstest
    
    # Shapiro-Wilk test (для небольших выборок)
    if len(z_scores) < 5000:
        shapiro_stat, shapiro_p = shapiro(z_scores)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = kstest(z_scores, 'norm')
    
    # Проверка coverage для разных confidence levels
    coverage_68 = np.mean(np.abs(z_scores) <= 1.0)  # 1 sigma
    coverage_95 = np.mean(np.abs(z_scores) <= 1.96)  # 2 sigma
    coverage_99 = np.mean(np.abs(z_scores) <= 2.58)  # 3 sigma
    
    metrics = {
        'expected_calibration_error': float(ece),
        'maximum_calibration_error': float(mce),
        'shapiro_statistic': float(shapiro_stat) if not np.isnan(shapiro_stat) else None,
        'shapiro_pvalue': float(shapiro_p) if not np.isnan(shapiro_p) else None,
        'ks_statistic': float(ks_stat),
        'ks_pvalue': float(ks_p),
        'coverage_68': float(coverage_68),
        'coverage_95': float(coverage_95),
        'coverage_99': float(coverage_99),
        'expected_coverage_68': 0.6827,
        'expected_coverage_95': 0.9545,
        'expected_coverage_99': 0.9973,
        'mean_z_score': float(np.mean(z_scores)),
        'std_z_score': float(np.std(z_scores))
    }
    
    logger.info(f"Calibration metrics: ECE={ece:.4f}, Coverage@95%={coverage_95:.3f}")
    
    return metrics


def compute_sharpness(uncertainties: np.ndarray) -> Dict[str, float]:
    """
    Вычисляет sharpness (насколько узкие доверительные интервалы).
    
    Хорошая модель должна быть одновременно calibrated (точные интервалы)
    и sharp (узкие интервалы).
    
    Args:
        uncertainties: Оценки uncertainty
    
    Returns:
        Dict[str, float]: Метрики sharpness
    """
    uncertainties = uncertainties.flatten()
    
    metrics = {
        'mean_uncertainty': float(np.mean(uncertainties)),
        'median_uncertainty': float(np.median(uncertainties)),
        'std_uncertainty': float(np.std(uncertainties)),
        'min_uncertainty': float(np.min(uncertainties)),
        'max_uncertainty': float(np.max(uncertainties)),
        'q25_uncertainty': float(np.percentile(uncertainties, 25)),
        'q75_uncertainty': float(np.percentile(uncertainties, 75)),
        'iqr_uncertainty': float(np.percentile(uncertainties, 75) - np.percentile(uncertainties, 25))
    }
    
    return metrics


def analyze_uncertainty_quality(predictions: np.ndarray,
                                uncertainties: np.ndarray,
                                targets: np.ndarray) -> Dict[str, any]:
    """
    Полный анализ качества uncertainty estimates.
    
    Args:
        predictions: Предсказания модели
        uncertainties: Оценки uncertainty
        targets: Истинные значения
    
    Returns:
        Dict: Полный набор метрик
    """
    calibration_metrics = compute_calibration_metrics(predictions, uncertainties, targets)
    sharpness_metrics = compute_sharpness(uncertainties)
    
    # Объединяем метрики
    quality_metrics = {
        'calibration': calibration_metrics,
        'sharpness': sharpness_metrics
    }
    
    # Общая оценка качества
    ece = calibration_metrics['expected_calibration_error']
    coverage_95 = calibration_metrics['coverage_95']
    expected_coverage_95 = calibration_metrics['expected_coverage_95']
    
    # Хорошая модель: ECE < 0.1 и coverage близко к ожидаемому
    is_well_calibrated = (ece < 0.1) and (abs(coverage_95 - expected_coverage_95) < 0.05)
    
    quality_metrics['overall'] = {
        'is_well_calibrated': is_well_calibrated,
        'calibration_score': float(1.0 - ece),  # Чем выше, тем лучше
        'coverage_deviation': float(abs(coverage_95 - expected_coverage_95))
    }
    
    logger.info(f"Uncertainty quality: {'GOOD' if is_well_calibrated else 'NEEDS IMPROVEMENT'}")
    
    return quality_metrics
