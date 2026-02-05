"""
Модуль для топологического анализа данных в молекулярных системах.

Содержит классы и функции для построения комплексов Вьеториса-Рипса,
вычисления персистентной гомологии и извлечения топологических признаков
для машинного обучения.
"""

from .vietoris_rips import (
    VietorisRipsComplex,
    MolecularRipsAnalyzer,
    create_rips_complex_from_molecule,
    batch_rips_analysis
)

from .persistence import (
    PersistentHomologyAnalyzer,
    analyze_molecular_persistence,
    batch_persistence_analysis
)

from .features import (
    TopologicalFeatureExtractor,
    TopologicalFeatureProcessor,
    extract_topological_features_from_molecules,
    create_topological_feature_pipeline
)

__all__ = [
    # Комплексы Вьеториса-Рипса
    'VietorisRipsComplex',
    'MolecularRipsAnalyzer',
    'create_rips_complex_from_molecule',
    'batch_rips_analysis',
    
    # Персистентная гомология
    'PersistentHomologyAnalyzer',
    'analyze_molecular_persistence',
    'batch_persistence_analysis',
    
    # Топологические признаки
    'TopologicalFeatureExtractor',
    'TopologicalFeatureProcessor',
    'extract_topological_features_from_molecules',
    'create_topological_feature_pipeline'
]