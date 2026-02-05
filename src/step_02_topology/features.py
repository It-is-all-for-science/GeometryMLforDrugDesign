"""
Модуль для извлечения топологических признаков из молекулярных структур.

Содержит функции для векторизации диаграмм персистентности и создания
признаков для машинного обучения.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union, Callable
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


class TopologicalFeatureExtractor:
    """
    Класс для извлечения топологических признаков из диаграмм персистентности.
    
    Поддерживает различные методы векторизации топологических данных
    для использования в машинном обучении.
    """
    
    def __init__(self, 
                 max_dimension: int = 2,
                 feature_types: List[str] = None):
        """
        Инициализация экстрактора топологических признаков.
        
        Args:
            max_dimension: Максимальная размерность для анализа
            feature_types: Типы признаков для извлечения
        """
        self.max_dimension = max_dimension
        
        if feature_types is None:
            self.feature_types = [
                'betti_numbers',
                'persistence_statistics',
                'persistence_landscapes',
                'persistence_images',
                'persistence_entropy'
            ]
        else:
            self.feature_types = feature_types
        
        logger.info(f"Инициализирован TopologicalFeatureExtractor: "
                   f"max_dimension={max_dimension}, "
                   f"feature_types={self.feature_types}")
    
    def extract_betti_numbers(self, diagrams: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Извлекает числа Бетти из диаграмм персистентности.
        
        Args:
            diagrams: Диаграммы персистентности
        
        Returns:
            np.ndarray: Вектор чисел Бетти [β₀, β₁, β₂, ...]
        """
        betti_numbers = np.zeros(self.max_dimension + 1)
        
        for dim in range(self.max_dimension + 1):
            if dim in diagrams:
                betti_numbers[dim] = len(diagrams[dim])
        
        return betti_numbers
    
    def extract_persistence_statistics(self, diagrams: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Извлекает статистические признаки из диаграмм персистентности.
        
        Args:
            diagrams: Диаграммы персистентности
        
        Returns:
            np.ndarray: Вектор статистических признаков
        """
        features = []
        
        for dim in range(self.max_dimension + 1):
            if dim in diagrams and len(diagrams[dim]) > 0:
                diagram = diagrams[dim]
                
                # Вычисляем персистентности
                births = diagram[:, 0]
                deaths = diagram[:, 1]
                persistences = deaths - births
                
                # Статистики рождения
                features.extend([
                    np.mean(births),
                    np.std(births),
                    np.min(births),
                    np.max(births)
                ])
                
                # Статистики смерти
                features.extend([
                    np.mean(deaths),
                    np.std(deaths),
                    np.min(deaths),
                    np.max(deaths)
                ])
                
                # Статистики персистентности
                features.extend([
                    np.mean(persistences),
                    np.std(persistences),
                    np.min(persistences),
                    np.max(persistences),
                    np.sum(persistences)  # Общая персистентность
                ])
                
            else:
                # Заполняем нулями для отсутствующих размерностей
                features.extend([0.0] * 13)  # 4 + 4 + 5 статистик
        
        return np.array(features)
    
    def extract_persistence_landscapes(self, 
                                     diagrams: Dict[int, np.ndarray],
                                     resolution: int = 50,
                                     num_landscapes: int = 5) -> np.ndarray:
        """
        Извлекает персистентные ландшафты.
        
        Args:
            diagrams: Диаграммы персистентности
            resolution: Разрешение дискретизации
            num_landscapes: Количество ландшафтов для каждой размерности
        
        Returns:
            np.ndarray: Векторизованные ландшафты
        """
        all_landscapes = []
        
        for dim in range(self.max_dimension + 1):
            if dim in diagrams and len(diagrams[dim]) > 0:
                diagram = diagrams[dim]
                landscapes = self._compute_landscapes(diagram, resolution, num_landscapes)
                all_landscapes.extend(landscapes.flatten())
            else:
                # Заполняем нулями для отсутствующих размерностей
                all_landscapes.extend([0.0] * (resolution * num_landscapes))
        
        return np.array(all_landscapes)
    
    def _compute_landscapes(self, 
                          diagram: np.ndarray,
                          resolution: int,
                          num_landscapes: int) -> np.ndarray:
        """
        Вычисляет персистентные ландшафты для одной размерности.
        
        Args:
            diagram: Диаграмма персистентности для одной размерности
            resolution: Разрешение дискретизации
            num_landscapes: Количество ландшафтов
        
        Returns:
            np.ndarray: Ландшафты [num_landscapes, resolution]
        """
        if len(diagram) == 0:
            return np.zeros((num_landscapes, resolution))
        
        # Определяем диапазон
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        
        min_val = np.min(births)
        max_val = np.max(deaths)
        
        # Создаем сетку
        x_grid = np.linspace(min_val, max_val, resolution)
        
        # Вычисляем функции ландшафта для каждого интервала
        landscape_functions = []
        
        for birth, death in diagram:
            # Треугольная функция
            midpoint = (birth + death) / 2
            height = (death - birth) / 2
            
            landscape_func = np.zeros(resolution)
            
            for i, x in enumerate(x_grid):
                if birth <= x <= death:
                    if x <= midpoint:
                        landscape_func[i] = height * (x - birth) / (midpoint - birth)
                    else:
                        landscape_func[i] = height * (death - x) / (death - midpoint)
            
            landscape_functions.append(landscape_func)
        
        # Вычисляем k-й ландшафт как k-й максимум в каждой точке
        landscapes = np.zeros((num_landscapes, resolution))
        
        if len(landscape_functions) > 0:
            landscape_matrix = np.array(landscape_functions)
            
            for i in range(resolution):
                # Сортируем значения в убывающем порядке
                sorted_values = np.sort(landscape_matrix[:, i])[::-1]
                
                # Берем первые num_landscapes значений
                for k in range(min(num_landscapes, len(sorted_values))):
                    landscapes[k, i] = sorted_values[k]
        
        return landscapes
    
    def extract_persistence_images(self, 
                                 diagrams: Dict[int, np.ndarray],
                                 resolution: int = 20,
                                 sigma: float = 1.0) -> np.ndarray:
        """
        Извлекает персистентные изображения.
        
        Args:
            diagrams: Диаграммы персистентности
            resolution: Разрешение изображения
            sigma: Параметр размытия Гаусса
        
        Returns:
            np.ndarray: Векторизованные персистентные изображения
        """
        all_images = []
        
        for dim in range(self.max_dimension + 1):
            if dim in diagrams and len(diagrams[dim]) > 0:
                diagram = diagrams[dim]
                image = self._compute_persistence_image(diagram, resolution, sigma)
                all_images.extend(image.flatten())
            else:
                # Заполняем нулями для отсутствующих размерностей
                all_images.extend([0.0] * (resolution * resolution))
        
        return np.array(all_images)
    
    def _compute_persistence_image(self, 
                                 diagram: np.ndarray,
                                 resolution: int,
                                 sigma: float) -> np.ndarray:
        """
        Вычисляет персистентное изображение для одной размерности.
        
        Args:
            diagram: Диаграмма персистентности
            resolution: Разрешение изображения
            sigma: Параметр размытия
        
        Returns:
            np.ndarray: Персистентное изображение [resolution, resolution]
        """
        if len(diagram) == 0:
            return np.zeros((resolution, resolution))
        
        # Преобразуем в координаты (birth, persistence)
        births = diagram[:, 0]
        deaths = diagram[:, 1]
        persistences = deaths - births
        
        # Определяем диапазоны
        birth_min, birth_max = np.min(births), np.max(births)
        pers_min, pers_max = 0, np.max(persistences)
        
        # Создаем сетку
        birth_grid = np.linspace(birth_min, birth_max, resolution)
        pers_grid = np.linspace(pers_min, pers_max, resolution)
        
        # Инициализируем изображение
        image = np.zeros((resolution, resolution))
        
        # Для каждой точки диаграммы добавляем Гауссово размытие
        for birth, persistence in zip(births, persistences):
            # Находим ближайшие индексы в сетке
            birth_idx = np.argmin(np.abs(birth_grid - birth))
            pers_idx = np.argmin(np.abs(pers_grid - persistence))
            
            # Добавляем Гауссово ядро
            for i in range(resolution):
                for j in range(resolution):
                    dist_sq = ((i - birth_idx) ** 2 + (j - pers_idx) ** 2)
                    weight = np.exp(-dist_sq / (2 * sigma ** 2))
                    image[i, j] += persistence * weight
        
        return image
    
    def extract_persistence_entropy(self, diagrams: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Извлекает энтропию персистентности.
        
        Args:
            diagrams: Диаграммы персистентности
        
        Returns:
            np.ndarray: Вектор энтропий для каждой размерности
        """
        entropies = []
        
        for dim in range(self.max_dimension + 1):
            if dim in diagrams and len(diagrams[dim]) > 0:
                diagram = diagrams[dim]
                
                # Вычисляем персистентности
                persistences = diagram[:, 1] - diagram[:, 0]
                
                # Нормализуем для получения вероятностей
                if np.sum(persistences) > 0:
                    probabilities = persistences / np.sum(persistences)
                    
                    # Вычисляем энтропию
                    entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                    entropies.append(entropy)
                else:
                    entropies.append(0.0)
            else:
                entropies.append(0.0)
        
        return np.array(entropies)
    
    def extract_all_features(self, diagrams: Dict[int, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Извлекает все типы топологических признаков.
        
        Args:
            diagrams: Диаграммы персистентности
        
        Returns:
            Dict[str, np.ndarray]: Словарь с различными типами признаков
        """
        features = {}
        
        if 'betti_numbers' in self.feature_types:
            features['betti_numbers'] = self.extract_betti_numbers(diagrams)
        
        if 'persistence_statistics' in self.feature_types:
            features['persistence_statistics'] = self.extract_persistence_statistics(diagrams)
        
        if 'persistence_landscapes' in self.feature_types:
            features['persistence_landscapes'] = self.extract_persistence_landscapes(diagrams)
        
        if 'persistence_images' in self.feature_types:
            features['persistence_images'] = self.extract_persistence_images(diagrams)
        
        if 'persistence_entropy' in self.feature_types:
            features['persistence_entropy'] = self.extract_persistence_entropy(diagrams)
        
        return features
    
    def concatenate_features(self, feature_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Объединяет различные типы признаков в один вектор.
        
        Args:
            feature_dict: Словарь с признаками
        
        Returns:
            np.ndarray: Объединенный вектор признаков
        """
        feature_vectors = []
        
        for feature_type in self.feature_types:
            if feature_type in feature_dict:
                feature_vectors.append(feature_dict[feature_type].flatten())
        
        if feature_vectors:
            return np.concatenate(feature_vectors)
        else:
            return np.array([])
    
    def extract_features(self, diagrams: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Извлекает топологические признаки из диаграмм персистентности.
        
        Args:
            diagrams: Диаграммы персистентности
        
        Returns:
            np.ndarray: Вектор топологических признаков
        """
        all_features = self.extract_all_features(diagrams)
        return self.concatenate_features(all_features)
    
    def get_feature_dimension(self) -> int:
        """
        Возвращает размерность вектора признаков.
        
        Returns:
            int: Размерность вектора признаков
        """
        # Вычисляем размерность на основе типов признаков
        total_dim = 0
        
        for feature_type in self.feature_types:
            if feature_type == 'betti_numbers':
                total_dim += self.max_dimension + 1
            elif feature_type == 'persistence_statistics':
                # Для каждой размерности: 4 статистики рождения + 4 смерти + 4 персистентности + 1 сумма
                total_dim += (self.max_dimension + 1) * 13
            elif feature_type == 'persistence_landscapes':
                # По умолчанию 50 точек на размерность
                total_dim += (self.max_dimension + 1) * 50
            elif feature_type == 'persistence_images':
                # По умолчанию 20x20 изображение на размерность
                total_dim += (self.max_dimension + 1) * 400
            elif feature_type == 'persistence_entropy':
                total_dim += self.max_dimension + 1
        
        return total_dim


class TopologicalFeatureProcessor:
    """
    Класс для обработки и нормализации топологических признаков.
    
    Включает методы для масштабирования, снижения размерности
    и подготовки признаков для машинного обучения.
    """
    
    def __init__(self):
        """Инициализация процессора признаков."""
        self.scaler = StandardScaler()
        self.pca = None
        self.is_fitted = False
        
    def fit_transform(self, 
                     feature_matrix: np.ndarray,
                     apply_pca: bool = False,
                     pca_components: Optional[int] = None) -> np.ndarray:
        """
        Обучает процессор и трансформирует признаки.
        
        Args:
            feature_matrix: Матрица признаков [n_samples, n_features]
            apply_pca: Применять ли PCA
            pca_components: Количество компонент PCA
        
        Returns:
            np.ndarray: Обработанные признаки
        """
        logger.info(f"Обучение процессора на {feature_matrix.shape[0]} образцах "
                   f"с {feature_matrix.shape[1]} признаками")
        
        # Стандартизация
        scaled_features = self.scaler.fit_transform(feature_matrix)
        
        # PCA (опционально)
        if apply_pca:
            if pca_components is None:
                # Автоматический выбор количества компонент (95% дисперсии)
                pca_components = min(feature_matrix.shape[0], feature_matrix.shape[1])
            
            self.pca = PCA(n_components=pca_components)
            processed_features = self.pca.fit_transform(scaled_features)
            
            logger.info(f"PCA: сохранено {self.pca.explained_variance_ratio_.sum():.3f} "
                       f"дисперсии в {pca_components} компонентах")
        else:
            processed_features = scaled_features
        
        self.is_fitted = True
        return processed_features
    
    def transform(self, feature_matrix: np.ndarray) -> np.ndarray:
        """
        Трансформирует новые признаки с использованием обученного процессора.
        
        Args:
            feature_matrix: Матрица признаков
        
        Returns:
            np.ndarray: Обработанные признаки
        """
        if not self.is_fitted:
            raise ValueError("Процессор не обучен. Вызовите fit_transform сначала.")
        
        # Стандартизация
        scaled_features = self.scaler.transform(feature_matrix)
        
        # PCA (если применялось)
        if self.pca is not None:
            processed_features = self.pca.transform(scaled_features)
        else:
            processed_features = scaled_features
        
        return processed_features
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Возвращает важность признаков (если применялось PCA).
        
        Returns:
            Optional[np.ndarray]: Важность признаков
        """
        if self.pca is not None:
            return self.pca.explained_variance_ratio_
        else:
            return None


def extract_topological_features_from_molecules(coordinates_list: List[torch.Tensor],
                                              atomic_numbers_list: Optional[List[torch.Tensor]] = None,
                                              feature_types: List[str] = None,
                                              max_dimension: int = 2) -> Tuple[np.ndarray, Dict[str, any]]:
    """
    Извлекает топологические признаки из списка молекул.
    
    Args:
        coordinates_list: Список координат молекул
        atomic_numbers_list: Список атомных номеров (опционально)
        feature_types: Типы признаков для извлечения
        max_dimension: Максимальная размерность
    
    Returns:
        Tuple[np.ndarray, Dict]: Матрица признаков и метаданные
    """
    from .persistence import batch_persistence_analysis
    
    logger.info(f"Извлечение топологических признаков из {len(coordinates_list)} молекул")
    
    # Вычисляем персистентную гомологию для всех молекул
    persistence_results = batch_persistence_analysis(
        coordinates_list, 
        atomic_numbers_list,
        max_dimension=max_dimension
    )
    
    # Инициализируем экстрактор признаков
    extractor = TopologicalFeatureExtractor(
        max_dimension=max_dimension,
        feature_types=feature_types
    )
    
    # Извлекаем признаки для каждой молекулы
    feature_matrices = []
    successful_extractions = 0
    
    for i, result in enumerate(persistence_results):
        if 'error' not in result:
            try:
                diagrams = result['diagrams']
                features = extractor.extract_all_features(diagrams)
                feature_vector = extractor.concatenate_features(features)
                
                feature_matrices.append(feature_vector)
                successful_extractions += 1
                
            except Exception as e:
                logger.error(f"Ошибка при извлечении признаков для молекулы {i}: {e}")
                # Добавляем нулевой вектор для сохранения индексации
                if len(feature_matrices) > 0:
                    feature_matrices.append(np.zeros_like(feature_matrices[0]))
                else:
                    # Если это первая молекула, создаем заглушку
                    dummy_features = extractor.extract_all_features({})
                    dummy_vector = extractor.concatenate_features(dummy_features)
                    feature_matrices.append(dummy_vector)
        else:
            logger.error(f"Пропускаем молекулу {i} из-за ошибки в персистентности")
            # Добавляем нулевой вектор
            if len(feature_matrices) > 0:
                feature_matrices.append(np.zeros_like(feature_matrices[0]))
    
    # Объединяем в матрицу
    if feature_matrices:
        feature_matrix = np.vstack(feature_matrices)
    else:
        feature_matrix = np.array([]).reshape(0, 0)
    
    # Метаданные
    metadata = {
        'num_molecules': len(coordinates_list),
        'successful_extractions': successful_extractions,
        'feature_types': extractor.feature_types,
        'max_dimension': max_dimension,
        'feature_matrix_shape': feature_matrix.shape,
        'extractor': extractor
    }
    
    logger.info(f"Извлечено {successful_extractions}/{len(coordinates_list)} "
               f"наборов признаков, размерность: {feature_matrix.shape}")
    
    return feature_matrix, metadata


def create_topological_feature_pipeline(feature_types: List[str] = None,
                                       max_dimension: int = 2,
                                       apply_pca: bool = False,
                                       pca_components: Optional[int] = None) -> Tuple[TopologicalFeatureExtractor, TopologicalFeatureProcessor]:
    """
    Создает полный пайплайн для извлечения и обработки топологических признаков.
    
    Args:
        feature_types: Типы признаков для извлечения
        max_dimension: Максимальная размерность
        apply_pca: Применять ли PCA
        pca_components: Количество компонент PCA
    
    Returns:
        Tuple: Экстрактор и процессор признаков
    """
    extractor = TopologicalFeatureExtractor(
        max_dimension=max_dimension,
        feature_types=feature_types
    )
    
    processor = TopologicalFeatureProcessor()
    
    logger.info("Создан пайплайн топологических признаков")
    
    return extractor, processor