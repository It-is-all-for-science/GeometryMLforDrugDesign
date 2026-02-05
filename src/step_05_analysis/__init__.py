"""
Модуль анализа результатов для молекулярного машинного обучения.

Содержит инструменты для сравнительного анализа моделей,
статистической оценки значимости различий и визуализации результатов.
"""

from .comparison import ComparisonAnalyzer, ModelResult, ComparisonResult

__all__ = [
    'ComparisonAnalyzer',
    'ModelResult', 
    'ComparisonResult'
]