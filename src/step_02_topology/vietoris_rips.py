"""
Модуль для построения комплексов Вьеториса-Рипса.

Содержит функции для создания симплициальных комплексов из молекулярных
структур с адаптивным выбором параметров для оптимальной производительности.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union
import gudhi as gd
from scipy.spatial.distance import pdist, squareform
import logging
from pathlib import Path
import pickle
import hashlib

logger = logging.getLogger(__name__)


class VietorisRipsComplex:
    """
    Класс для построения комплексов Вьеториса-Рипса из молекулярных данных.
    
    Оптимизирован для работы с молекулярными структурами и включает
    кэширование для повышения производительности.
    """
    
    def __init__(self, 
                 max_edge_length: float = 10.0,
                 max_dimension: int = 2,
                 cache_dir: Optional[str] = "data/topology_cache"):
        """
        Инициализация построителя комплексов Вьеториса-Рипса.
        
        Args:
            max_edge_length: Максимальная длина ребра в комплексе (Å)
            max_dimension: Максимальная размерность симплексов
            cache_dir: Директория для кэширования результатов
        """
        self.max_edge_length = max_edge_length
        self.max_dimension = max_dimension
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Инициализирован VietorisRipsComplex: "
                   f"max_edge_length={max_edge_length}, "
                   f"max_dimension={max_dimension}")
    
    def compute_distance_matrix(self, coordinates: torch.Tensor) -> np.ndarray:
        """
        Вычисляет матрицу расстояний между атомами.
        
        Args:
            coordinates: Координаты атомов [N, 3]
        
        Returns:
            np.ndarray: Матрица расстояний [N, N]
        """
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()
        
        # Используем scipy для эффективного вычисления расстояний
        distances = pdist(coordinates, metric='euclidean')
        distance_matrix = squareform(distances)
        
        return distance_matrix
    
    def build_rips_complex(self, 
                          coordinates: torch.Tensor,
                          max_edge_length: Optional[float] = None) -> gd.RipsComplex:
        """
        Строит комплекс Вьеториса-Рипса из координат атомов.
        
        Args:
            coordinates: Координаты атомов [N, 3]
            max_edge_length: Максимальная длина ребра (если None, использует self.max_edge_length)
        
        Returns:
            gudhi.RipsComplex: Построенный комплекс
        """
        if max_edge_length is None:
            max_edge_length = self.max_edge_length
        
        # Проверяем кэш
        cache_key = self._get_cache_key(coordinates, max_edge_length)
        if self.cache_dir and self._is_cached(cache_key):
            logger.debug(f"Загружаем комплекс из кэша: {cache_key}")
            return self._load_from_cache(cache_key)
        
        # Вычисляем матрицу расстояний
        distance_matrix = self.compute_distance_matrix(coordinates)
        
        # Строим комплекс Вьеториса-Рипса
        logger.debug(f"Строим Rips комплекс для {coordinates.shape[0]} точек")
        
        rips_complex = gd.RipsComplex(
            distance_matrix=distance_matrix,
            max_edge_length=max_edge_length
        )
        
        # Создаем симплициальный комплекс
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=self.max_dimension)
        
        # Сохраняем в кэш
        if self.cache_dir:
            self._save_to_cache(cache_key, simplex_tree)
        
        logger.debug(f"Построен комплекс с {simplex_tree.num_simplices()} симплексами")
        
        return simplex_tree
    
    def adaptive_max_edge_length(self, coordinates: torch.Tensor) -> float:
        """
        Автоматически выбирает оптимальную максимальную длину ребра.
        
        Использует статистику расстояний между атомами для выбора
        разумного порога, который захватывает локальную структуру
        без создания слишком плотного комплекса.
        
        Args:
            coordinates: Координаты атомов [N, 3]
        
        Returns:
            float: Оптимальная максимальная длина ребра
        """
        distance_matrix = self.compute_distance_matrix(coordinates)
        
        # Исключаем диагональ (нулевые расстояния)
        distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        
        # Используем статистику для выбора порога
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Эвристика: среднее + 1 стандартное отклонение
        # Это захватывает большинство ближайших соседей
        adaptive_threshold = mean_dist + std_dist
        
        # Ограничиваем разумными пределами
        min_threshold = np.percentile(distances, 10)  # 10-й процентиль
        max_threshold = np.percentile(distances, 90)  # 90-й процентиль
        
        adaptive_threshold = np.clip(adaptive_threshold, min_threshold, max_threshold)
        
        logger.debug(f"Адаптивный порог: {adaptive_threshold:.3f} Å "
                    f"(mean={mean_dist:.3f}, std={std_dist:.3f})")
        
        return adaptive_threshold
    
    def compute_multiple_complexes(self, 
                                 coordinates: torch.Tensor,
                                 edge_lengths: Optional[List[float]] = None) -> List[gd.SimplexTree]:
        """
        Вычисляет комплексы для нескольких значений максимальной длины ребра.
        
        Полезно для анализа персистентности на разных масштабах.
        
        Args:
            coordinates: Координаты атомов [N, 3]
            edge_lengths: Список максимальных длин ребер
        
        Returns:
            List[gudhi.SimplexTree]: Список комплексов
        """
        if edge_lengths is None:
            # Автоматически генерируем последовательность порогов
            adaptive_length = self.adaptive_max_edge_length(coordinates)
            edge_lengths = np.linspace(0.5, adaptive_length * 2, 10).tolist()
        
        complexes = []
        
        for edge_length in edge_lengths:
            logger.debug(f"Строим комплекс с max_edge_length={edge_length:.3f}")
            complex_tree = self.build_rips_complex(coordinates, edge_length)
            complexes.append(complex_tree)
        
        return complexes
    
    def get_complex_statistics(self, simplex_tree: gd.SimplexTree) -> Dict[str, int]:
        """
        Вычисляет статистики симплициального комплекса.
        
        Args:
            simplex_tree: Симплициальный комплекс
        
        Returns:
            Dict[str, int]: Статистики комплекса
        """
        stats = {
            'num_vertices': simplex_tree.num_vertices(),
            'num_simplices': simplex_tree.num_simplices(),
        }
        
        # Подсчитываем симплексы по размерностям
        for dim in range(self.max_dimension + 1):
            count = sum(1 for simplex, _ in simplex_tree.get_simplices() 
                       if len(simplex) - 1 == dim)
            stats[f'num_{dim}_simplices'] = count
        
        return stats
    
    def _get_cache_key(self, coordinates: torch.Tensor, max_edge_length: float) -> str:
        """Генерирует ключ для кэширования."""
        # Создаем хэш от координат и параметров
        coords_bytes = coordinates.detach().cpu().numpy().tobytes()
        params_str = f"{max_edge_length}_{self.max_dimension}"
        
        hash_input = coords_bytes + params_str.encode()
        cache_key = hashlib.md5(hash_input).hexdigest()
        
        return cache_key
    
    def _is_cached(self, cache_key: str) -> bool:
        """Проверяет наличие результата в кэше."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        return cache_file.exists()
    
    def _save_to_cache(self, cache_key: str, simplex_tree: gd.SimplexTree) -> None:
        """Сохраняет результат в кэш."""
        try:
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(simplex_tree, f)
            logger.debug(f"Сохранен в кэш: {cache_file}")
        except Exception as e:
            logger.warning(f"Не удалось сохранить в кэш: {e}")
    
    def _load_from_cache(self, cache_key: str) -> gd.SimplexTree:
        """Загружает результат из кэша."""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        with open(cache_file, 'rb') as f:
            return pickle.load(f)


class MolecularRipsAnalyzer:
    """
    Специализированный анализатор комплексов Вьеториса-Рипса для молекул.
    
    Учитывает химическую специфику молекулярных структур.
    """
    
    def __init__(self):
        """Инициализация молекулярного анализатора."""
        # Типичные длины связей в органических молекулах (Å)
        self.bond_lengths = {
            'C-C': 1.54,
            'C=C': 1.34,
            'C≡C': 1.20,
            'C-N': 1.47,
            'C=N': 1.29,
            'C-O': 1.43,
            'C=O': 1.23,
            'N-N': 1.45,
            'N-O': 1.40,
            'O-O': 1.48,
            'C-H': 1.09,
            'N-H': 1.01,
            'O-H': 0.96
        }
        
        # Ван-дер-ваальсовы радиусы (Å)
        self.vdw_radii = {
            1: 1.20,   # H
            6: 1.70,   # C
            7: 1.55,   # N
            8: 1.52,   # O
            9: 1.47,   # F
            15: 1.80,  # P
            16: 1.80,  # S
        }
    
    def get_molecular_edge_threshold(self, 
                                   coordinates: torch.Tensor,
                                   atomic_numbers: Optional[torch.Tensor] = None) -> float:
        """
        Вычисляет химически обоснованный порог для молекулярных комплексов.
        
        Args:
            coordinates: Координаты атомов [N, 3]
            atomic_numbers: Атомные номера [N] (опционально)
        
        Returns:
            float: Рекомендуемый порог для максимальной длины ребра
        """
        distance_matrix = VietorisRipsComplex().compute_distance_matrix(coordinates)
        
        # Исключаем диагональ
        distances = distance_matrix[np.triu_indices_from(distance_matrix, k=1)]
        
        # Если есть информация об атомах, используем химические знания
        if atomic_numbers is not None:
            # Находим типичные длины связей
            typical_bond_length = 1.5  # Средняя длина C-C связи
            
            # Ищем расстояния, соответствующие химическим связям
            bond_distances = distances[distances < 2.0]  # Предполагаем связи < 2 Å
            
            if len(bond_distances) > 0:
                max_bond_length = np.max(bond_distances)
                # Добавляем буфер для ван-дер-ваальсовых взаимодействий
                threshold = max_bond_length + 2.0
            else:
                # Если связи не найдены, используем эвристику
                threshold = np.percentile(distances, 20)  # 20-й процентиль
        else:
            # Без химической информации используем статистический подход
            threshold = np.percentile(distances, 30)  # 30-й процентиль
        
        # Ограничиваем разумными пределами для молекул
        threshold = np.clip(threshold, 1.0, 8.0)
        
        logger.debug(f"Молекулярный порог: {threshold:.3f} Å")
        
        return threshold
    
    def analyze_molecular_cavities(self, 
                                 coordinates: torch.Tensor,
                                 atomic_numbers: Optional[torch.Tensor] = None) -> Dict[str, any]:
        """
        Анализирует молекулярные полости с помощью топологии.
        
        Args:
            coordinates: Координаты атомов [N, 3]
            atomic_numbers: Атомные номера [N] (опционально)
        
        Returns:
            Dict: Результаты анализа полостей
        """
        # Используем специальный порог для анализа полостей
        cavity_threshold = self.get_molecular_edge_threshold(coordinates, atomic_numbers)
        
        # Строим комплекс
        rips_builder = VietorisRipsComplex(max_edge_length=cavity_threshold)
        simplex_tree = rips_builder.build_rips_complex(coordinates)
        
        # Анализируем топологические особенности
        stats = rips_builder.get_complex_statistics(simplex_tree)
        
        # Оцениваем наличие полостей
        # 1-мерные дыры могут указывать на кольцевые структуры
        # 2-мерные дыры могут указывать на полости
        
        analysis = {
            'threshold_used': cavity_threshold,
            'complex_stats': stats,
            'has_rings': stats.get('num_1_simplices', 0) > stats.get('num_vertices', 0),
            'potential_cavities': stats.get('num_2_simplices', 0) > 0,
            'complexity_score': stats.get('num_simplices', 0) / max(stats.get('num_vertices', 1), 1)
        }
        
        return analysis
    
    def compare_molecular_topologies(self, 
                                   molecules_coords: List[torch.Tensor]) -> Dict[str, any]:
        """
        Сравнивает топологические характеристики нескольких молекул.
        
        Args:
            molecules_coords: Список координат молекул
        
        Returns:
            Dict: Результаты сравнительного анализа
        """
        analyses = []
        
        for i, coords in enumerate(molecules_coords):
            logger.debug(f"Анализируем молекулу {i+1}/{len(molecules_coords)}")
            analysis = self.analyze_molecular_cavities(coords)
            analyses.append(analysis)
        
        # Сравнительная статистика
        complexity_scores = [a['complexity_score'] for a in analyses]
        thresholds = [a['threshold_used'] for a in analyses]
        
        comparison = {
            'num_molecules': len(molecules_coords),
            'individual_analyses': analyses,
            'complexity_stats': {
                'mean': np.mean(complexity_scores),
                'std': np.std(complexity_scores),
                'min': np.min(complexity_scores),
                'max': np.max(complexity_scores)
            },
            'threshold_stats': {
                'mean': np.mean(thresholds),
                'std': np.std(thresholds),
                'min': np.min(thresholds),
                'max': np.max(thresholds)
            }
        }
        
        return comparison


def create_rips_complex_from_molecule(coordinates: torch.Tensor,
                                    atomic_numbers: Optional[torch.Tensor] = None,
                                    adaptive_threshold: bool = True,
                                    max_dimension: int = 2) -> Tuple[gd.SimplexTree, Dict[str, any]]:
    """
    Удобная функция для создания комплекса Вьеториса-Рипса из молекулы.
    
    Args:
        coordinates: Координаты атомов [N, 3]
        atomic_numbers: Атомные номера [N] (опционально)
        adaptive_threshold: Использовать ли адаптивный порог
        max_dimension: Максимальная размерность симплексов
    
    Returns:
        Tuple[gudhi.SimplexTree, Dict]: Комплекс и метаданные
    """
    # Выбираем подходящий порог
    if adaptive_threshold:
        if atomic_numbers is not None:
            analyzer = MolecularRipsAnalyzer()
            threshold = analyzer.get_molecular_edge_threshold(coordinates, atomic_numbers)
        else:
            builder = VietorisRipsComplex(max_dimension=max_dimension)
            threshold = builder.adaptive_max_edge_length(coordinates)
    else:
        threshold = 5.0  # Значение по умолчанию
    
    # Строим комплекс
    builder = VietorisRipsComplex(
        max_edge_length=threshold,
        max_dimension=max_dimension
    )
    
    simplex_tree = builder.build_rips_complex(coordinates)
    stats = builder.get_complex_statistics(simplex_tree)
    
    metadata = {
        'threshold_used': threshold,
        'adaptive_threshold': adaptive_threshold,
        'max_dimension': max_dimension,
        'num_atoms': coordinates.shape[0],
        'stats': stats
    }
    
    return simplex_tree, metadata


def batch_rips_analysis(coordinates_list: List[torch.Tensor],
                       atomic_numbers_list: Optional[List[torch.Tensor]] = None,
                       max_workers: int = 4) -> List[Tuple[gd.SimplexTree, Dict[str, any]]]:
    """
    Параллельный анализ комплексов Вьеториса-Рипса для множества молекул.
    
    Args:
        coordinates_list: Список координат молекул
        atomic_numbers_list: Список атомных номеров (опционально)
        max_workers: Максимальное количество параллельных процессов
    
    Returns:
        List[Tuple]: Список комплексов и метаданных
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    results = []
    
    # Для Windows лучше использовать последовательную обработку
    # из-за особенностей multiprocessing
    logger.info(f"Обрабатываем {len(coordinates_list)} молекул последовательно")
    
    for i, coords in enumerate(coordinates_list):
        atomic_nums = atomic_numbers_list[i] if atomic_numbers_list else None
        
        try:
            result = create_rips_complex_from_molecule(coords, atomic_nums)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Обработано {i + 1}/{len(coordinates_list)} молекул")
                
        except Exception as e:
            logger.error(f"Ошибка при обработке молекулы {i}: {e}")
            results.append((None, {'error': str(e)}))
    
    logger.info(f"Завершена обработка {len(coordinates_list)} молекул")
    
    return results