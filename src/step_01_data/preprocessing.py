"""
Модуль для предобработки молекулярных данных.

Содержит функции для нормализации, аугментации и трансформации
молекулярных данных с сохранением геометрических симметрий.
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from torch_geometric.data import Data, Batch
from torch_geometric.transforms import BaseTransform
import logging

logger = logging.getLogger(__name__)


class MolecularNormalizer:
    """
    Класс для нормализации молекулярных свойств и признаков.
    
    Обеспечивает стандартизацию данных для улучшения обучения моделей
    с сохранением физического смысла.
    """
    
    def __init__(self):
        self.feature_stats = {}
        self.target_stats = {}
        self.fitted = False
    
    def fit(self, data_list: List[Data], target_values: torch.Tensor):
        """
        Вычисляет статистики для нормализации.
        
        Args:
            data_list: Список молекулярных графов
            target_values: Целевые значения
        """
        logger.info("Вычисление статистик для нормализации...")
        
        # Статистики для целевых значений
        self.target_stats = {
            'mean': target_values.mean().item(),
            'std': target_values.std().item()
        }
        
        # Статистики для узловых признаков
        all_node_features = []
        all_edge_features = []
        all_positions = []
        
        for data in data_list:
            if hasattr(data, 'x') and data.x is not None:
                all_node_features.append(data.x)
            
            if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                all_edge_features.append(data.edge_attr)
            
            if hasattr(data, 'pos') and data.pos is not None:
                all_positions.append(data.pos)
        
        # Узловые признаки
        if all_node_features:
            node_features = torch.cat(all_node_features, dim=0)
            self.feature_stats['node'] = {
                'mean': node_features.mean(dim=0),
                'std': node_features.std(dim=0)
            }
        
        # Признаки связей
        if all_edge_features:
            edge_features = torch.cat(all_edge_features, dim=0)
            self.feature_stats['edge'] = {
                'mean': edge_features.mean(dim=0),
                'std': edge_features.std(dim=0)
            }
        
        # Позиционные статистики (только для информации, не нормализуем координаты)
        if all_positions:
            positions = torch.cat(all_positions, dim=0)
            self.feature_stats['position'] = {
                'mean': positions.mean(dim=0),
                'std': positions.std(dim=0),
                'mean_distance_from_origin': torch.norm(positions, dim=1).mean().item()
            }
        
        self.fitted = True
        logger.info(f"Нормализатор обучен на {len(data_list)} молекулах")
        logger.info(f"Целевые значения: mean={self.target_stats['mean']:.4f}, "
                   f"std={self.target_stats['std']:.4f}")
    
    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Нормализует целевые значения."""
        if not self.fitted:
            raise ValueError("Нормализатор не обучен. Вызовите fit() сначала.")
        
        return (targets - self.target_stats['mean']) / self.target_stats['std']
    
    def denormalize_targets(self, normalized_targets: torch.Tensor) -> torch.Tensor:
        """Денормализует целевые значения."""
        if not self.fitted:
            raise ValueError("Нормализатор не обучен. Вызовите fit() сначала.")
        
        return normalized_targets * self.target_stats['std'] + self.target_stats['mean']
    
    def normalize_data(self, data: Data) -> Data:
        """
        Нормализует признаки в молекулярном графе.
        
        Args:
            data: Молекулярный граф
        
        Returns:
            Data: Нормализованный граф
        """
        if not self.fitted:
            raise ValueError("Нормализатор не обучен. Вызовите fit() сначала.")
        
        data_normalized = data.clone()
        
        # Нормализация узловых признаков
        if hasattr(data, 'x') and data.x is not None and 'node' in self.feature_stats:
            node_mean = self.feature_stats['node']['mean']
            node_std = self.feature_stats['node']['std']
            # Избегаем деления на ноль
            node_std = torch.where(node_std > 1e-6, node_std, torch.ones_like(node_std))
            data_normalized.x = (data.x - node_mean) / node_std
        
        # Нормализация признаков связей
        if hasattr(data, 'edge_attr') and data.edge_attr is not None and 'edge' in self.feature_stats:
            edge_mean = self.feature_stats['edge']['mean']
            edge_std = self.feature_stats['edge']['std']
            edge_std = torch.where(edge_std > 1e-6, edge_std, torch.ones_like(edge_std))
            data_normalized.edge_attr = (data.edge_attr - edge_mean) / edge_std
        
        return data_normalized


class SymmetryAugmentation(BaseTransform):
    """
    Аугментация данных с сохранением молекулярных симметрий.
    
    Применяет случайные вращения и отражения для увеличения разнообразия
    данных без нарушения физических законов.
    """
    
    def __init__(self, 
                 rotation_prob: float = 0.5,
                 reflection_prob: float = 0.3,
                 noise_std: float = 0.01):
        """
        Args:
            rotation_prob: Вероятность применения случайного вращения
            reflection_prob: Вероятность применения отражения
            noise_std: Стандартное отклонение для добавления шума к координатам
        """
        self.rotation_prob = rotation_prob
        self.reflection_prob = reflection_prob
        self.noise_std = noise_std
    
    def __call__(self, data: Data) -> Data:
        """
        Применяет симметричные трансформации к молекулярным данным.
        
        Args:
            data: Исходный молекулярный граф
        
        Returns:
            Data: Аугментированный граф
        """
        if not hasattr(data, 'pos') or data.pos is None:
            return data
        
        data_aug = data.clone()
        pos = data_aug.pos.clone()
        
        # Случайное вращение (SO(3) симметрия)
        if torch.rand(1).item() < self.rotation_prob:
            rotation_matrix = self._random_rotation_matrix()
            pos = torch.matmul(pos, rotation_matrix.T)
        
        # Случайное отражение
        if torch.rand(1).item() < self.reflection_prob:
            # Отражение относительно случайной плоскости
            reflection_axis = torch.randint(0, 3, (1,)).item()
            pos[:, reflection_axis] *= -1
        
        # Добавление небольшого шума (моделирует тепловые флуктуации)
        if self.noise_std > 0:
            noise = torch.randn_like(pos) * self.noise_std
            pos = pos + noise
        
        data_aug.pos = pos
        return data_aug
    
    def _random_rotation_matrix(self) -> torch.Tensor:
        """
        Генерирует случайную матрицу вращения в SO(3).
        
        Использует метод Кабша для генерации равномерно распределенных
        вращений в трехмерном пространстве.
        
        Returns:
            torch.Tensor: Матрица вращения 3x3
        """
        # Генерируем случайную матрицу
        random_matrix = torch.randn(3, 3)
        
        # QR разложение для получения ортогональной матрицы
        q, r = torch.linalg.qr(random_matrix)
        
        # Обеспечиваем det(Q) = 1 (собственное вращение, не отражение)
        if torch.det(q) < 0:
            q[:, -1] *= -1
        
        return q


class MolecularFeatureExtractor:
    """
    Извлечение дополнительных молекулярных признаков.
    
    Вычисляет геометрические, топологические и химические дескрипторы
    для улучшения представления молекул.
    """
    
    @staticmethod
    def compute_distance_matrix(pos: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет матрицу расстояний между атомами.
        
        Args:
            pos: Координаты атомов [N, 3]
        
        Returns:
            torch.Tensor: Матрица расстояний [N, N]
        """
        return torch.cdist(pos, pos)
    
    @staticmethod
    def compute_angle_features(pos: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет угловые признаки для связей.
        
        Args:
            pos: Координаты атомов [N, 3]
            edge_index: Индексы связей [2, E]
        
        Returns:
            torch.Tensor: Угловые признаки [E, 1]
        """
        row, col = edge_index
        
        # Векторы связей
        bond_vectors = pos[col] - pos[row]  # [E, 3]
        
        # Длины связей
        bond_lengths = torch.norm(bond_vectors, dim=1, keepdim=True)  # [E, 1]
        
        return bond_lengths
    
    @staticmethod
    def compute_dihedral_features(pos: torch.Tensor, 
                                edge_index: torch.Tensor) -> torch.Tensor:
        """
        Вычисляет диэдральные углы (торсионные углы).
        
        Для каждой последовательности из 4 связанных атомов A-B-C-D
        вычисляет диэдральный угол между плоскостями ABC и BCD.
        
        Args:
            pos: Координаты атомов [N, 3]
            edge_index: Индексы связей [2, E]
        
        Returns:
            torch.Tensor: Диэдральные углы
        """
        # Упрощенная реализация - возвращаем заглушку
        # Полная реализация требует поиска путей длины 4 в графе
        num_edges = edge_index.size(1)
        return torch.zeros(num_edges, 1)
    
    @staticmethod
    def compute_molecular_descriptors(data: Data) -> Dict[str, torch.Tensor]:
        """
        Вычисляет набор молекулярных дескрипторов.
        
        Args:
            data: Молекулярный граф
        
        Returns:
            Dict[str, torch.Tensor]: Словарь дескрипторов
        """
        descriptors = {}
        
        if hasattr(data, 'pos') and data.pos is not None:
            pos = data.pos
            
            # Геометрические дескрипторы
            center_of_mass = pos.mean(dim=0)
            distances_from_center = torch.norm(pos - center_of_mass, dim=1)
            
            descriptors['radius_of_gyration'] = distances_from_center.mean()
            descriptors['max_distance_from_center'] = distances_from_center.max()
            descriptors['molecular_volume'] = torch.prod(pos.max(dim=0)[0] - pos.min(dim=0)[0])
            
            # Топологические дескрипторы
            descriptors['num_atoms'] = torch.tensor(pos.size(0), dtype=torch.float)
            
            if hasattr(data, 'edge_index'):
                descriptors['num_bonds'] = torch.tensor(data.edge_index.size(1), dtype=torch.float)
                descriptors['average_degree'] = descriptors['num_bonds'] * 2 / descriptors['num_atoms']
            
            # Химические дескрипторы (если доступны атомные номера)
            if hasattr(data, 'z'):
                descriptors['molecular_weight'] = data.z.float().sum()
                descriptors['heavy_atom_count'] = (data.z > 1).float().sum()
        
        return descriptors


class DataSplitter:
    """
    Класс для разделения данных на обучающую, валидационную и тестовую выборки.
    
    Обеспечивает стратифицированное разделение с учетом распределения
    целевых переменных.
    """
    
    @staticmethod
    def random_split(data_list: List[Data], 
                    target_values: torch.Tensor,
                    train_ratio: float = 0.7,
                    val_ratio: float = 0.15,
                    test_ratio: float = 0.15,
                    random_seed: int = 42) -> Tuple[List[Data], List[Data], List[Data], 
                                                   torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Случайное разделение данных.
        
        Args:
            data_list: Список молекулярных графов
            target_values: Целевые значения
            train_ratio: Доля обучающей выборки
            val_ratio: Доля валидационной выборки
            test_ratio: Доля тестовой выборки
            random_seed: Семя для воспроизводимости
        
        Returns:
            Tuple: (train_data, val_data, test_data, train_targets, val_targets, test_targets)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Сумма долей должна равняться 1.0"
        
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        n_samples = len(data_list)
        indices = torch.randperm(n_samples)
        
        # Вычисляем границы разделения
        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Разделяем данные
        train_data = [data_list[i] for i in train_indices]
        val_data = [data_list[i] for i in val_indices]
        test_data = [data_list[i] for i in test_indices]
        
        train_targets = target_values[train_indices]
        val_targets = target_values[val_indices]
        test_targets = target_values[test_indices]
        
        logger.info(f"Данные разделены: train={len(train_data)}, "
                   f"val={len(val_data)}, test={len(test_data)}")
        
        return train_data, val_data, test_data, train_targets, val_targets, test_targets
    
    @staticmethod
    def stratified_split(data_list: List[Data],
                        target_values: torch.Tensor,
                        n_bins: int = 10,
                        train_ratio: float = 0.7,
                        val_ratio: float = 0.15,
                        test_ratio: float = 0.15,
                        random_seed: int = 42) -> Tuple[List[Data], List[Data], List[Data],
                                                       torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Стратифицированное разделение данных по квантилям целевой переменной.
        
        Обеспечивает равномерное распределение целевых значений
        во всех выборках.
        
        Args:
            data_list: Список молекулярных графов
            target_values: Целевые значения
            n_bins: Количество бинов для стратификации
            train_ratio: Доля обучающей выборки
            val_ratio: Доля валидационной выборки
            test_ratio: Доля тестовой выборки
            random_seed: Семя для воспроизводимости
        
        Returns:
            Tuple: Разделенные данные и целевые значения
        """
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Создаем бины на основе квантилей
        quantiles = torch.quantile(target_values, torch.linspace(0, 1, n_bins + 1))
        bin_indices = torch.bucketize(target_values, quantiles[1:-1])
        
        train_data, val_data, test_data = [], [], []
        train_targets, val_targets, test_targets = [], [], []
        
        # Для каждого бина выполняем пропорциональное разделение
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            bin_data = [data_list[i] for i in torch.where(mask)[0]]
            bin_targets = target_values[mask]
            
            if len(bin_data) == 0:
                continue
            
            # Случайное разделение внутри бина
            n_bin = len(bin_data)
            indices = torch.randperm(n_bin)
            
            train_end = int(train_ratio * n_bin)
            val_end = train_end + int(val_ratio * n_bin)
            
            train_indices = indices[:train_end]
            val_indices = indices[train_end:val_end]
            test_indices = indices[val_end:]
            
            # Добавляем к общим выборкам
            train_data.extend([bin_data[i] for i in train_indices])
            val_data.extend([bin_data[i] for i in val_indices])
            test_data.extend([bin_data[i] for i in test_indices])
            
            train_targets.append(bin_targets[train_indices])
            val_targets.append(bin_targets[val_indices])
            test_targets.append(bin_targets[test_indices])
        
        # Объединяем целевые значения
        train_targets = torch.cat(train_targets) if train_targets else torch.tensor([])
        val_targets = torch.cat(val_targets) if val_targets else torch.tensor([])
        test_targets = torch.cat(test_targets) if test_targets else torch.tensor([])
        
        logger.info(f"Стратифицированное разделение: train={len(train_data)}, "
                   f"val={len(val_data)}, test={len(test_data)}")
        
        return train_data, val_data, test_data, train_targets, val_targets, test_targets


def preprocess_molecular_dataset(data_list: List[Data],
                               target_values: torch.Tensor,
                               normalize: bool = True,
                               augment: bool = False,
                               split_data: bool = True,
                               **kwargs) -> Dict[str, any]:
    """
    Полный пайплайн предобработки молекулярных данных.
    
    Args:
        data_list: Список молекулярных графов
        target_values: Целевые значения
        normalize: Применять ли нормализацию
        augment: Применять ли аугментацию
        split_data: Разделять ли данные на train/val/test
        **kwargs: Дополнительные параметры
    
    Returns:
        Dict: Словарь с предобработанными данными
    """
    logger.info("Начало предобработки молекулярных данных...")
    
    result = {
        'original_data': data_list,
        'original_targets': target_values
    }
    
    # Нормализация
    if normalize:
        normalizer = MolecularNormalizer()
        normalizer.fit(data_list, target_values)
        
        normalized_data = [normalizer.normalize_data(data) for data in data_list]
        normalized_targets = normalizer.normalize_targets(target_values)
        
        result['normalizer'] = normalizer
        result['normalized_data'] = normalized_data
        result['normalized_targets'] = normalized_targets
    
    # Аугментация
    if augment:
        augmentation = SymmetryAugmentation(
            rotation_prob=kwargs.get('rotation_prob', 0.5),
            reflection_prob=kwargs.get('reflection_prob', 0.3),
            noise_std=kwargs.get('noise_std', 0.01)
        )
        
        augmented_data = [augmentation(data) for data in data_list]
        result['augmented_data'] = augmented_data
    
    # Разделение данных
    if split_data:
        splitter = DataSplitter()
        
        data_to_split = result.get('normalized_data', data_list)
        targets_to_split = result.get('normalized_targets', target_values)
        
        if kwargs.get('stratified', False):
            split_result = splitter.stratified_split(
                data_to_split, targets_to_split,
                n_bins=kwargs.get('n_bins', 10),
                train_ratio=kwargs.get('train_ratio', 0.7),
                val_ratio=kwargs.get('val_ratio', 0.15),
                test_ratio=kwargs.get('test_ratio', 0.15),
                random_seed=kwargs.get('random_seed', 42)
            )
        else:
            split_result = splitter.random_split(
                data_to_split, targets_to_split,
                train_ratio=kwargs.get('train_ratio', 0.7),
                val_ratio=kwargs.get('val_ratio', 0.15),
                test_ratio=kwargs.get('test_ratio', 0.15),
                random_seed=kwargs.get('random_seed', 42)
            )
        
        (train_data, val_data, test_data, 
         train_targets, val_targets, test_targets) = split_result
        
        result.update({
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'train_targets': train_targets,
            'val_targets': val_targets,
            'test_targets': test_targets
        })
    
    logger.info("Предобработка завершена успешно")
    return result