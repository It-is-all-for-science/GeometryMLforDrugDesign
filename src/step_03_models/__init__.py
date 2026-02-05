"""
Модуль для реализации геометрических нейронных сетей.

Содержит реализации E(n) Equivariant Graph Neural Networks (EGNN)
и других геометрических моделей для молекулярного машинного обучения.
"""

from .egnn import (
    EGNNLayer,
    EGNNModel,
    EGNNConfig,
    create_egnn_model,
    test_equivariance
)

from .baseline import (
    FCNNBaseline,
    GCNBaseline,
    TabularBaseline,
    BaselineConfig,
    create_baseline_model
)

from .utils import (
    ModelUtils,
    EquivarianceTest,
    GeometricTransforms
)

__all__ = [
    # EGNN модели
    'EGNNLayer',
    'EGNNModel', 
    'EGNNConfig',
    'create_egnn_model',
    'test_equivariance',
    
    # Baseline модели
    'FCNNBaseline',
    'GCNBaseline',
    'TabularBaseline',
    'BaselineConfig',
    'create_baseline_model',
    
    # Утилиты
    'ModelUtils',
    'EquivarianceTest',
    'GeometricTransforms'
]