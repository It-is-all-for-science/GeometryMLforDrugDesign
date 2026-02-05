"""
Модуль для работы с молекулярными данными.

Содержит классы и функции для загрузки, предобработки и визуализации
молекулярных датасетов с сохранением геометрических симметрий.
"""

from .loaders import (
    MolecularDataLoader,
    MolecularData,
    ProteinComplexData,
    GeometryPreservingTransform,
    create_molecular_dataloader
)

from .preprocessing import (
    MolecularNormalizer,
    SymmetryAugmentation,
    MolecularFeatureExtractor,
    DataSplitter,
    preprocess_molecular_dataset
)

from .visualization import (
    MolecularVisualizer,
    create_comprehensive_visualization_report
)

__all__ = [
    # Загрузка данных
    'MolecularDataLoader',
    'MolecularData',
    'ProteinComplexData',
    'GeometryPreservingTransform',
    'create_molecular_dataloader',
    
    # Предобработка
    'MolecularNormalizer',
    'SymmetryAugmentation',
    'MolecularFeatureExtractor',
    'DataSplitter',
    'preprocess_molecular_dataset',
    
    # Визуализация
    'MolecularVisualizer',
    'create_comprehensive_visualization_report'
]