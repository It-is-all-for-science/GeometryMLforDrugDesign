"""
Модуль для загрузки молекулярных датасетов.

Этот модуль содержит классы для загрузки и предобработки различных
молекулярных датасетов, включая QM9, MD17 и PDBbind, с сохранением
геометрических симметрий.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union
from dataclasses import dataclass
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import QM9
from torch_geometric.transforms import Compose
import requests
import tarfile
import gzip
from pathlib import Path
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MolecularData:
    """Структура данных для представления молекулярной информации."""
    atomic_numbers: torch.Tensor      # [N] атомные номера
    coordinates: torch.Tensor         # [N, 3] 3D координаты
    edge_indices: torch.Tensor        # [2, E] связи между атомами
    edge_attributes: torch.Tensor     # [E, F] признаки связей
    molecular_properties: torch.Tensor # [P] целевые свойства
    topology_features: Optional[torch.Tensor] = None  # [T] топологические признаки


@dataclass
class ProteinComplexData:
    """Структура данных для белок-лигандных комплексов."""
    protein_coords: torch.Tensor      # [N_prot, 3] координаты белка
    ligand_coords: torch.Tensor       # [N_lig, 3] координаты лиганда
    protein_features: torch.Tensor    # [N_prot, F] признаки остатков
    ligand_features: torch.Tensor     # [N_lig, F] признаки лиганда
    interface_edges: torch.Tensor     # [2, E_int] межмолекулярные связи
    binding_affinity: torch.Tensor    # скаляр - аффинность связывания
    interface_topology: Optional[torch.Tensor] = None  # топология интерфейса


class MolecularDataLoader:
    """
    Основной класс для загрузки и предобработки молекулярных данных.
    
    Поддерживает загрузку QM9, MD17, PDBbind датасетов с сохранением
    геометрических симметрий и инвариантов.
    """
    
    def __init__(self, data_root: str = "data/raw"):
        """
        Инициализация загрузчика данных.
        
        Args:
            data_root: Корневая директория для хранения данных
        """
        self.data_root = Path(data_root)
        self.data_root.mkdir(parents=True, exist_ok=True)
        
        # Создаем поддиректории для разных датасетов
        (self.data_root / "qm9").mkdir(exist_ok=True)
        (self.data_root / "md17").mkdir(exist_ok=True)
        (self.data_root / "pdbbind").mkdir(exist_ok=True)
        
        logger.info(f"Инициализирован MolecularDataLoader с корневой директорией: {self.data_root}")
    
    def load_qm9(self, target_property: str = "homo_lumo_gap") -> Tuple[List[Data], torch.Tensor, Dict[str, any]]:
        """
        Загружает датасет QM9 с молекулярными свойствами.
        
        QM9 содержит ~134k малых органических молекул с квантово-химическими
        свойствами, вычисленными на уровне DFT.
        
        Args:
            target_property: Целевое свойство для предсказания
                           ('homo_lumo_gap', 'energy', 'dipole_moment', etc.)
        
        Returns:
            Tuple[List[Data], torch.Tensor, Dict]: 
                - Список молекулярных графов
                - Тензор целевых значений
                - Словарь с метаданными
        """
        logger.info(f"Загрузка QM9 датасета с целевым свойством: {target_property}")
        
        try:
            # Загружаем QM9 через PyTorch Geometric
            dataset = QM9(root=str(self.data_root / "qm9"))
            
            # Маппинг названий свойств к индексам в QM9
            property_mapping = {
                'dipole_moment': 0,     # mu (D)
                'polarizability': 1,    # alpha (Bohr^3)
                'homo': 2,              # HOMO (eV)
                'lumo': 3,              # LUMO (eV)
                'homo_lumo_gap': 4,     # gap (eV)
                'electronic_spatial_extent': 5,  # R^2 (Bohr^2)
                'zero_point_vibrational_energy': 6,  # ZPVE (eV)
                'internal_energy_0k': 7,        # U0 (eV)
                'internal_energy_298k': 8,      # U (eV)
                'enthalpy_298k': 9,             # H (eV)
                'free_energy_298k': 10,         # G (eV)
                'heat_capacity': 11,            # Cv (cal/mol/K)
            }
            
            if target_property not in property_mapping:
                raise ValueError(f"Неизвестное свойство: {target_property}. "
                               f"Доступные: {list(property_mapping.keys())}")
            
            target_idx = property_mapping[target_property]
            
            # Извлекаем данные
            molecular_graphs = []
            target_values = []
            
            for i, data in enumerate(dataset):
                if i % 10000 == 0:
                    logger.info(f"Обработано {i}/{len(dataset)} молекул")
                
                # Проверяем валидность данных и размер тензора y
                if (data.y is not None and 
                    data.y.numel() > target_idx and 
                    not torch.isnan(data.y.flatten()[target_idx])):
                    molecular_graphs.append(data)
                    target_values.append(data.y.flatten()[target_idx])
                elif i < 10:  # Логируем первые несколько проблемных случаев
                    logger.warning(f"Пропускаем молекулу {i}: "
                                 f"y.shape={data.y.shape if data.y is not None else None}, "
                                 f"target_idx={target_idx}")
                    if data.y is not None:
                        logger.warning(f"  y.flatten()={data.y.flatten()[:5]}...")  # Первые 5 значений
            
            if not target_values:
                raise ValueError(f"Не найдено валидных данных для свойства {target_property}")
            
            target_tensor = torch.stack(target_values)
            
            # Метаданные
            metadata = {
                'dataset_name': 'QM9',
                'target_property': target_property,
                'num_molecules': len(molecular_graphs),
                'target_mean': target_tensor.mean().item(),
                'target_std': target_tensor.std().item(),
                'target_min': target_tensor.min().item(),
                'target_max': target_tensor.max().item(),
                'property_units': self._get_qm9_units(target_property)
            }
            
            logger.info(f"Загружено {len(molecular_graphs)} молекул из QM9")
            logger.info(f"Статистика {target_property}: "
                       f"mean={metadata['target_mean']:.4f}, "
                       f"std={metadata['target_std']:.4f}")
            
            return molecular_graphs, target_tensor, metadata
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке QM9: {e}")
            raise
    
    def load_md17(self, molecule: str = "benzene") -> Tuple[List[Data], torch.Tensor, Dict[str, any]]:
        """
        Загружает датасет MD17 с молекулярной динамикой.
        
        MD17 содержит траектории молекулярной динамики для небольших
        органических молекул с энергиями и силами на уровне DFT.
        
        Args:
            molecule: Название молекулы ('benzene', 'toluene', 'malonaldehyde', etc.)
        
        Returns:
            Tuple[List[Data], torch.Tensor, Dict]: Данные MD17
        """
        logger.info(f"Загрузка MD17 для молекулы: {molecule}")
        
        # Доступные молекулы в MD17
        available_molecules = [
            'benzene', 'toluene', 'malonaldehyde', 'salicylic_acid',
            'aspirin', 'ethanol', 'uracil', 'naphthalene'
        ]
        
        if molecule not in available_molecules:
            raise ValueError(f"Молекула {molecule} недоступна. "
                           f"Доступные: {available_molecules}")
        
        # Пока возвращаем заглушку - полная реализация MD17 требует дополнительной работы
        logger.warning("MD17 загрузчик находится в разработке. Возвращаем заглушку.")
        
        # Создаем простую заглушку
        dummy_data = Data(
            x=torch.randn(10, 5),  # 10 атомов, 5 признаков
            pos=torch.randn(10, 3),  # 3D координаты
            edge_index=torch.randint(0, 10, (2, 20)),  # 20 связей
            y=torch.randn(1)  # энергия
        )
        
        return [dummy_data], torch.randn(1), {
            'dataset_name': 'MD17',
            'molecule': molecule,
            'num_conformations': 1,
            'note': 'Заглушка - требует полной реализации'
        }
    
    def load_pdbbind(self, subset: str = "core", use_mock_data: bool = True) -> Tuple[List[ProteinComplexData], torch.Tensor, Dict[str, any]]:
        """
        Загружает датасет PDBbind с белок-лигандными комплексами.
        
        PDBbind содержит структуры белок-лигандных комплексов с экспериментально
        измеренными аффинностями связывания.
        
        Args:
            subset: Подмножество данных ('core', 'refined', 'general')
            use_mock_data: Использовать синтетические данные для тестирования
        
        Returns:
            Tuple[List[ProteinComplexData], torch.Tensor, Dict]: Данные PDBbind
        """
        logger.info(f"Загрузка PDBbind, подмножество: {subset}")
        
        if use_mock_data:
            # Используем синтетические данные для тестирования
            from .pdb_parser import create_mock_pdbbind_data
            
            num_complexes = {
                'core': 195,      # Реальный размер core set
                'refined': 4057,  # Реальный размер refined set  
                'general': 17679  # Реальный размер general set
            }.get(subset, 100)
            
            return create_mock_pdbbind_data(num_complexes)
        
        else:
            # Реальная загрузка PDBbind данных
            pdbbind_dir = self.data_root / "pdbbind"
            
            if not pdbbind_dir.exists():
                logger.warning(f"Директория PDBbind не найдена: {pdbbind_dir}")
                logger.info("Используем синтетические данные вместо реальных")
                from .pdb_parser import create_mock_pdbbind_data
                return create_mock_pdbbind_data(100)
            
            # Здесь будет реальная загрузка PDB файлов
            # Пока возвращаем синтетические данные
            logger.warning("Реальная загрузка PDBbind в разработке. Используем синтетические данные.")
            from .pdb_parser import create_mock_pdbbind_data
            return create_mock_pdbbind_data(100)
    
    def get_molecular_properties(self, data_list: List[Data]) -> Dict[str, torch.Tensor]:
        """
        Извлекает различные молекулярные свойства из списка данных.
        
        Args:
            data_list: Список молекулярных графов
        
        Returns:
            Dict[str, torch.Tensor]: Словарь с различными свойствами
        """
        properties = {
            'num_atoms': [],
            'num_bonds': [],
            'molecular_weight': [],
            'center_of_mass': [],
            'radius_of_gyration': []
        }
        
        for data in data_list:
            # Количество атомов
            properties['num_atoms'].append(data.x.size(0))
            
            # Количество связей
            properties['num_bonds'].append(data.edge_index.size(1))
            
            # Приблизительная молекулярная масса (сумма атомных номеров)
            if hasattr(data, 'z'):
                properties['molecular_weight'].append(data.z.float().sum())
            else:
                properties['molecular_weight'].append(torch.tensor(0.0))
            
            # Центр масс
            if hasattr(data, 'pos'):
                properties['center_of_mass'].append(data.pos.mean(dim=0))
            else:
                properties['center_of_mass'].append(torch.zeros(3))
            
            # Радиус гирации
            if hasattr(data, 'pos'):
                center = data.pos.mean(dim=0, keepdim=True)
                distances = torch.norm(data.pos - center, dim=1)
                properties['radius_of_gyration'].append(distances.mean())
            else:
                properties['radius_of_gyration'].append(torch.tensor(0.0))
        
        # Конвертируем в тензоры
        for key in properties:
            if key == 'center_of_mass':
                properties[key] = torch.stack(properties[key])
            else:
                properties[key] = torch.tensor(properties[key])
        
        return properties
    
    def extract_binding_interface(self, 
                                protein_coords: torch.Tensor, 
                                ligand_coords: torch.Tensor,
                                cutoff: float = 5.0) -> torch.Tensor:
        """
        Извлекает интерфейс связывания между белком и лигандом.
        
        Args:
            protein_coords: Координаты атомов белка [N_prot, 3]
            ligand_coords: Координаты атомов лиганда [N_lig, 3]
            cutoff: Пороговое расстояние для определения интерфейса (Å)
        
        Returns:
            torch.Tensor: Индексы связей интерфейса [2, E_interface]
        """
        # Вычисляем расстояния между всеми парами атомов
        distances = torch.cdist(protein_coords, ligand_coords)  # [N_prot, N_lig]
        
        # Находим пары атомов в пределах cutoff
        protein_indices, ligand_indices = torch.where(distances < cutoff)
        
        # Смещаем индексы лиганда на количество атомов белка
        ligand_indices_shifted = ligand_indices + protein_coords.size(0)
        
        # Создаем тензор связей
        interface_edges = torch.stack([protein_indices, ligand_indices_shifted])
        
        logger.info(f"Найдено {interface_edges.size(1)} межмолекулярных связей "
                   f"в интерфейсе (cutoff={cutoff} Å)")
        
        return interface_edges
    
    def _get_qm9_units(self, property_name: str) -> str:
        """Возвращает единицы измерения для свойств QM9."""
        units_mapping = {
            'dipole_moment': 'D (Debye)',
            'polarizability': 'Bohr³',
            'homo': 'eV',
            'lumo': 'eV',
            'homo_lumo_gap': 'eV',
            'electronic_spatial_extent': 'Bohr²',
            'zero_point_vibrational_energy': 'eV',
            'internal_energy_0k': 'eV',
            'internal_energy_298k': 'eV',
            'enthalpy_298k': 'eV',
            'free_energy_298k': 'eV',
            'heat_capacity': 'cal/mol/K'
        }
        return units_mapping.get(property_name, 'unknown')


class GeometryPreservingTransform:
    """
    Класс для трансформаций данных с сохранением геометрических симметрий.
    
    Обеспечивает трансляционную и вращательную инвариантность/эквивариантность
    при предобработке молекулярных данных.
    """
    
    def __init__(self, center_coordinates: bool = True, normalize_positions: bool = False):
        """
        Args:
            center_coordinates: Центрировать координаты относительно центра масс
            normalize_positions: Нормализовать позиции по радиусу гирации
        """
        self.center_coordinates = center_coordinates
        self.normalize_positions = normalize_positions
    
    def __call__(self, data: Data) -> Data:
        """
        Применяет геометрически инвариантные трансформации.
        
        Args:
            data: Молекулярный граф
        
        Returns:
            Data: Трансформированный граф
        """
        if not hasattr(data, 'pos') or data.pos is None:
            return data
        
        pos = data.pos.clone()
        
        # Центрирование координат (трансляционная инвариантность)
        if self.center_coordinates:
            center_of_mass = pos.mean(dim=0, keepdim=True)
            pos = pos - center_of_mass
        
        # Нормализация по радиусу гирации (масштабная инвариантность)
        if self.normalize_positions:
            radius_of_gyration = torch.norm(pos, dim=1).mean()
            if radius_of_gyration > 1e-6:  # избегаем деления на ноль
                pos = pos / radius_of_gyration
        
        # Создаем новый объект данных с трансформированными координатами
        data_transformed = data.clone()
        data_transformed.pos = pos
        
        return data_transformed


def create_molecular_dataloader(dataset_name: str, 
                              target_property: str = None,
                              batch_size: int = 32,
                              shuffle: bool = True,
                              **kwargs) -> torch.utils.data.DataLoader:
    """
    Создает DataLoader для молекулярных данных.
    
    Args:
        dataset_name: Название датасета ('qm9', 'md17', 'pdbbind')
        target_property: Целевое свойство для предсказания
        batch_size: Размер батча
        shuffle: Перемешивать ли данные
        **kwargs: Дополнительные аргументы для загрузчика
    
    Returns:
        DataLoader: Настроенный загрузчик данных
    """
    loader = MolecularDataLoader()
    
    if dataset_name.lower() == 'qm9':
        data_list, targets, metadata = loader.load_qm9(target_property or 'homo_lumo_gap')
    elif dataset_name.lower() == 'md17':
        molecule = kwargs.get('molecule', 'benzene')
        data_list, targets, metadata = loader.load_md17(molecule)
    elif dataset_name.lower() == 'pdbbind':
        subset = kwargs.get('subset', 'core')
        data_list, targets, metadata = loader.load_pdbbind(subset)
    else:
        raise ValueError(f"Неизвестный датасет: {dataset_name}")
    
    # Применяем геометрически инвариантные трансформации
    transform = GeometryPreservingTransform(
        center_coordinates=kwargs.get('center_coordinates', True),
        normalize_positions=kwargs.get('normalize_positions', False)
    )
    
    transformed_data = [transform(data) for data in data_list]
    
    # Создаем DataLoader
    from torch_geometric.loader import DataLoader
    
    dataloader = DataLoader(
        transformed_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=kwargs.get('num_workers', 0)
    )
    
    logger.info(f"Создан DataLoader для {dataset_name} с {len(transformed_data)} образцами")
    
    return dataloader, metadata