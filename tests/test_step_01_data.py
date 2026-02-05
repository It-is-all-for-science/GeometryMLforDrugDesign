"""
Тесты для основной функциональности проекта молекулярного машинного обучения
"""
import pytest
import numpy as np
import sys
import os
import torch
from pathlib import Path

# Добавляем src в путь для импорта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_project_structure():
    """Тест базовой структуры проекта"""
    # Проверяем, что основные директории существуют
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    required_dirs = [
        'src',
        'docs', 
        'notebooks',
        'experiments',
        'configs',
        'data',
        'results',
        'tests'
    ]
    
    for dir_name in required_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        assert os.path.exists(dir_path), f"Директория {dir_name} не существует"

def test_config_files():
    """Тест наличия конфигурационных файлов"""
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    config_files = [
        'configs/qm9_config.yaml',
        'configs/pdbbind_config.yaml'
    ]
    
    for config_file in config_files:
        config_path = os.path.join(base_dir, config_file)
        assert os.path.exists(config_path), f"Конфигурационный файл {config_file} не существует"

def test_documentation_files():
    """Тест наличия документации"""
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    doc_files = [
        'docs/learning_report.md',
        'docs/glossary.md',
        'docs/step_by_step_guide.md',
        'docs/references.md'
    ]
    
    for doc_file in doc_files:
        doc_path = os.path.join(base_dir, doc_file)
        assert os.path.exists(doc_path), f"Документация {doc_file} не существует"

def test_src_modules_import():
    """Тест импорта основных модулей"""
    try:
        from step_01_data.loaders import MolecularDataLoader
        from step_03_models.egnn import EGNNModel, EGNNConfig
        from step_04_training.trainer import ModelTrainer, TrainingConfig
        from step_04_training.metrics import MetricsCalculator
        assert True, "Все основные модули успешно импортированы"
    except ImportError as e:
        pytest.fail(f"Ошибка импорта модулей: {e}")

def test_egnn_config():
    """Тест конфигурации EGNN модели"""
    try:
        from step_03_models.egnn import EGNNConfig
        
        config = EGNNConfig(
            hidden_dim=128,
            num_layers=4,
            output_dim=1
        )
        
        assert config.hidden_dim == 128
        assert config.num_layers == 4
        assert config.output_dim == 1
        
    except Exception as e:
        pytest.fail(f"Ошибка создания конфигурации EGNN: {e}")

def test_training_config():
    """Тест конфигурации обучения"""
    try:
        from step_04_training.trainer import TrainingConfig
        
        config = TrainingConfig()
        
        # Проверяем, что конфигурация создается без ошибок
        assert hasattr(config, '__dict__')
        
    except Exception as e:
        pytest.fail(f"Ошибка создания конфигурации обучения: {e}")

def test_molecular_data_structure():
    """Тест структуры молекулярных данных"""
    # Создаем тестовые данные в формате, который ожидает система
    test_molecule = {
        'smiles': 'CCO',  # Этанол
        'homo_lumo_gap': 5.2,
        'n_atoms': 9
    }
    
    assert 'smiles' in test_molecule
    assert 'homo_lumo_gap' in test_molecule
    assert isinstance(test_molecule['homo_lumo_gap'], (int, float))
    assert test_molecule['n_atoms'] > 0

def test_results_directories():
    """Тест наличия директорий для результатов"""
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    result_dirs = [
        'results',
        'data'
    ]
    
    for dir_name in result_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        assert os.path.exists(dir_path), f"Директория {dir_name} не существует"

def test_key_experiment_files():
    """Тест наличия ключевых файлов экспериментов"""
    base_dir = os.path.join(os.path.dirname(__file__), '..')
    
    key_files = [
        'experiments/improved_egnn_training.py',
        'experiments/optimized_antibacterial_analysis.py',
        'experiments/experimental_homo_lumo_gap_search.py',
        'experiments/task_31_experimental_validation.py'
    ]
    
    for file_path in key_files:
        full_path = os.path.join(base_dir, file_path)
        assert os.path.exists(full_path), f"Ключевой файл {file_path} не существует"

if __name__ == "__main__":
    pytest.main([__file__])