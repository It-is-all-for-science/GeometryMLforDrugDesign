"""
Модуль для вычисления персистентной гомологии.

Содержит функции для анализа топологических признаков на разных масштабах
и создания диаграмм персистентности для молекулярных структур.
"""

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Union
import gudhi as gd
import matplotlib.pyplot as plt
import seaborn as sns
from ripser import ripser
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class PersistentHomologyAnalyzer:
    """
    Класс для анализа персистентной гомологии молекулярных структур.
    
    Поддерживает различные методы вычисления и визуализации
    диаграмм персистентности.
    """
    
    def __init__(self, 
                 max_dimension: int = 2,
                 method: str = 'gudhi'):
        """
        Инициализация анализатора персистентной гомологии.
        
        Args:
            max_dimension: Максимальная размерность для анализа
            method: Метод вычисления ('gudhi' или 'ripser')
        """
        self.max_dimension = max_dimension
        self.method = method
        
        logger.info(f"Инициализирован PersistentHomologyAnalyzer: "
                   f"max_dimension={max_dimension}, method={method}")
    
    def compute_persistence_gudhi(self, 
                                simplex_tree: gd.SimplexTree) -> List[Tuple[int, Tuple[float, float]]]:
        """
        Вычисляет персистентную гомологию с помощью GUDHI.
        
        Args:
            simplex_tree: Симплициальный комплекс
        
        Returns:
            List[Tuple]: Диаграмма персистентности
        """
        logger.debug("Вычисляем персистентную гомологию с GUDHI")
        
        # Вычисляем персистентную гомологию
        persistence = simplex_tree.persistence()
        
        # Фильтруем бесконечные интервалы и форматируем результат
        formatted_persistence = []
        
        for interval in persistence:
            dimension = interval[0]
            birth_death = interval[1]
            
            # Пропускаем бесконечные интервалы для упрощения
            if len(birth_death) == 2 and birth_death[1] != float('inf'):
                formatted_persistence.append((dimension, birth_death))
        
        logger.debug(f"Найдено {len(formatted_persistence)} конечных интервалов персистентности")
        
        return formatted_persistence
    
    def compute_persistence_ripser(self, 
                                 distance_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Вычисляет персистентную гомологию с помощью Ripser.
        
        Args:
            distance_matrix: Матрица расстояний
        
        Returns:
            Dict[str, np.ndarray]: Диаграммы персистентности по размерностям
        """
        logger.debug("Вычисляем персистентную гомологию с Ripser")
        
        # Вычисляем с помощью Ripser
        result = ripser(
            distance_matrix, 
            maxdim=self.max_dimension,
            distance_matrix=True
        )
        
        return result['dgms']
    
    def compute_persistence_from_coordinates(self, 
                                           coordinates: torch.Tensor,
                                           max_edge_length: Optional[float] = None) -> Dict[str, any]:
        """
        Вычисляет персистентную гомологию напрямую из координат.
        
        Args:
            coordinates: Координаты точек [N, 3]
            max_edge_length: Максимальная длина ребра
        
        Returns:
            Dict: Результаты анализа персистентности
        """
        if isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()
        
        if self.method == 'ripser':
            # Используем Ripser для прямого анализа
            from scipy.spatial.distance import pdist, squareform
            
            distances = pdist(coordinates)
            distance_matrix = squareform(distances)
            
            diagrams = self.compute_persistence_ripser(distance_matrix)
            
            result = {
                'method': 'ripser',
                'diagrams': diagrams,
                'coordinates': coordinates,
                'num_points': len(coordinates)
            }
            
        else:  # gudhi
            # Используем GUDHI через комплекс Вьеториса-Рипса
            from .vietoris_rips import VietorisRipsComplex
            
            if max_edge_length is None:
                builder = VietorisRipsComplex(max_dimension=self.max_dimension)
                max_edge_length = builder.adaptive_max_edge_length(torch.from_numpy(coordinates))
            
            builder = VietorisRipsComplex(
                max_edge_length=max_edge_length,
                max_dimension=self.max_dimension
            )
            
            simplex_tree = builder.build_rips_complex(torch.from_numpy(coordinates))
            persistence = self.compute_persistence_gudhi(simplex_tree)
            
            # Конвертируем в формат, совместимый с Ripser
            diagrams = self._convert_gudhi_to_ripser_format(persistence)
            
            result = {
                'method': 'gudhi',
                'diagrams': diagrams,
                'persistence': persistence,
                'simplex_tree': simplex_tree,
                'coordinates': coordinates,
                'num_points': len(coordinates),
                'max_edge_length': max_edge_length
            }
        
        # Добавляем статистики
        result['statistics'] = self._compute_persistence_statistics(diagrams)
        
        return result
    
    def _convert_gudhi_to_ripser_format(self, 
                                      persistence: List[Tuple[int, Tuple[float, float]]]) -> Dict[int, np.ndarray]:
        """
        Конвертирует формат GUDHI в формат Ripser для совместимости.
        
        Args:
            persistence: Персистентность в формате GUDHI
        
        Returns:
            Dict[int, np.ndarray]: Диаграммы в формате Ripser
        """
        diagrams = {}
        
        # Группируем по размерностям
        for dimension, (birth, death) in persistence:
            if dimension not in diagrams:
                diagrams[dimension] = []
            diagrams[dimension].append([birth, death])
        
        # Конвертируем в numpy массивы
        for dim in diagrams:
            diagrams[dim] = np.array(diagrams[dim])
        
        return diagrams
    
    def _compute_persistence_statistics(self, 
                                      diagrams: Dict[int, np.ndarray]) -> Dict[str, any]:
        """
        Вычисляет статистики диаграмм персистентности.
        
        Args:
            diagrams: Диаграммы персистентности
        
        Returns:
            Dict: Статистики
        """
        stats = {
            'dimensions': list(diagrams.keys()),
            'total_features': sum(len(dgm) for dgm in diagrams.values())
        }
        
        for dim, diagram in diagrams.items():
            if len(diagram) > 0:
                # Вычисляем персистентности (death - birth)
                persistences = diagram[:, 1] - diagram[:, 0]
                
                stats[f'dim_{dim}'] = {
                    'num_features': len(diagram),
                    'max_persistence': float(np.max(persistences)),
                    'mean_persistence': float(np.mean(persistences)),
                    'std_persistence': float(np.std(persistences)),
                    'total_persistence': float(np.sum(persistences))
                }
            else:
                stats[f'dim_{dim}'] = {
                    'num_features': 0,
                    'max_persistence': 0.0,
                    'mean_persistence': 0.0,
                    'std_persistence': 0.0,
                    'total_persistence': 0.0
                }
        
        return stats
    
    def plot_persistence_diagram(self, 
                               diagrams: Dict[int, np.ndarray],
                               title: str = "Диаграмма персистентности",
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Создает визуализацию диаграммы персистентности.
        
        Args:
            diagrams: Диаграммы персистентности
            title: Заголовок графика
            figsize: Размер фигуры
        
        Returns:
            matplotlib.figure.Figure: График диаграммы
        """
        # Определяем количество размерностей для subplot'ов
        dimensions = sorted(diagrams.keys())
        n_dims = len(dimensions)
        
        if n_dims == 0:
            fig, ax = plt.subplots(figsize=figsize)
            ax.text(0.5, 0.5, 'Нет данных для отображения', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return fig
        
        # Создаем subplot'ы
        fig, axes = plt.subplots(1, n_dims, figsize=figsize)
        if n_dims == 1:
            axes = [axes]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for i, dim in enumerate(dimensions):
            ax = axes[i]
            diagram = diagrams[dim]
            
            if len(diagram) > 0:
                # Основная диаграмма
                births = diagram[:, 0]
                deaths = diagram[:, 1]
                
                ax.scatter(births, deaths, 
                          c=colors[dim % len(colors)], 
                          alpha=0.7, s=50,
                          label=f'H_{dim}')
                
                # Диагональная линия (birth = death)
                max_val = max(np.max(births), np.max(deaths))
                min_val = min(np.min(births), np.min(deaths))
                
                ax.plot([min_val, max_val], [min_val, max_val], 
                       'k--', alpha=0.5, linewidth=1)
                
                # Настройка осей
                ax.set_xlabel('Birth')
                ax.set_ylabel('Death')
                ax.set_title(f'H_{dim} (размерность {dim})')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Добавляем статистику как текст
                persistences = deaths - births
                stats_text = f'Признаков: {len(diagram)}\n'
                stats_text += f'Макс. персистентность: {np.max(persistences):.3f}\n'
                stats_text += f'Средняя: {np.mean(persistences):.3f}'
                
                ax.text(0.02, 0.98, stats_text, 
                       transform=ax.transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            else:
                ax.text(0.5, 0.5, f'Нет признаков H_{dim}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'H_{dim} (размерность {dim})')
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        return fig
    
    def plot_persistence_barcode(self, 
                               diagrams: Dict[int, np.ndarray],
                               title: str = "Штрих-код персистентности",
                               figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Создает штрих-код персистентности.
        
        Args:
            diagrams: Диаграммы персистентности
            title: Заголовок графика
            figsize: Размер фигуры
        
        Returns:
            matplotlib.figure.Figure: График штрих-кода
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        y_pos = 0
        
        for dim in sorted(diagrams.keys()):
            diagram = diagrams[dim]
            color = colors[dim % len(colors)]
            
            for birth, death in diagram:
                ax.plot([birth, death], [y_pos, y_pos], 
                       color=color, linewidth=3, alpha=0.7)
                y_pos += 1
            
            # Добавляем разделитель между размерностями
            if len(diagram) > 0:
                y_pos += 2
        
        ax.set_xlabel('Параметр фильтрации')
        ax.set_ylabel('Топологические признаки')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Легенда
        legend_elements = []
        for dim in sorted(diagrams.keys()):
            if len(diagrams[dim]) > 0:
                legend_elements.append(
                    plt.Line2D([0], [0], color=colors[dim % len(colors)], 
                             linewidth=3, label=f'H_{dim}')
                )
        
        if legend_elements:
            ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        return fig
    
    def compute_persistence_landscapes(self, 
                                     diagrams: Dict[int, np.ndarray],
                                     resolution: int = 100) -> Dict[int, np.ndarray]:
        """
        Вычисляет персистентные ландшафты для векторизации топологических признаков.
        
        Args:
            diagrams: Диаграммы персистентности
            resolution: Разрешение для дискретизации
        
        Returns:
            Dict[int, np.ndarray]: Ландшафты по размерностям
        """
        landscapes = {}
        
        for dim, diagram in diagrams.items():
            if len(diagram) == 0:
                landscapes[dim] = np.zeros(resolution)
                continue
            
            # Определяем диапазон для ландшафта
            births = diagram[:, 0]
            deaths = diagram[:, 1]
            
            min_val = np.min(births)
            max_val = np.max(deaths)
            
            # Создаем сетку
            x_grid = np.linspace(min_val, max_val, resolution)
            landscape = np.zeros(resolution)
            
            # Вычисляем ландшафт как максимум треугольных функций
            for birth, death in diagram:
                # Треугольная функция для интервала [birth, death]
                midpoint = (birth + death) / 2
                
                for i, x in enumerate(x_grid):
                    if birth <= x <= death:
                        if x <= midpoint:
                            # Возрастающая часть
                            value = (x - birth) / (midpoint - birth) * (death - birth) / 2
                        else:
                            # Убывающая часть
                            value = (death - x) / (death - midpoint) * (death - birth) / 2
                        
                        landscape[i] = max(landscape[i], value)
            
            landscapes[dim] = landscape
        
        return landscapes
    
    def compare_persistence_diagrams(self, 
                                   diagrams1: Dict[int, np.ndarray],
                                   diagrams2: Dict[int, np.ndarray]) -> Dict[str, float]:
        """
        Сравнивает две диаграммы персистентности.
        
        Args:
            diagrams1: Первая диаграмма
            diagrams2: Вторая диаграмма
        
        Returns:
            Dict[str, float]: Метрики сходства
        """
        try:
            import persim
            
            comparison = {}
            
            # Сравниваем по размерностям
            all_dims = set(diagrams1.keys()) | set(diagrams2.keys())
            
            for dim in all_dims:
                dgm1 = diagrams1.get(dim, np.array([]).reshape(0, 2))
                dgm2 = diagrams2.get(dim, np.array([]).reshape(0, 2))
                
                if len(dgm1) == 0 and len(dgm2) == 0:
                    comparison[f'wasserstein_dim_{dim}'] = 0.0
                    comparison[f'bottleneck_dim_{dim}'] = 0.0
                else:
                    # Расстояние Вассерштейна
                    try:
                        wasserstein_dist = persim.wasserstein(dgm1, dgm2)
                        comparison[f'wasserstein_dim_{dim}'] = wasserstein_dist
                    except:
                        comparison[f'wasserstein_dim_{dim}'] = float('inf')
                    
                    # Расстояние бутылочного горлышка
                    try:
                        bottleneck_dist = persim.bottleneck(dgm1, dgm2)
                        comparison[f'bottleneck_dim_{dim}'] = bottleneck_dist
                    except:
                        comparison[f'bottleneck_dim_{dim}'] = float('inf')
            
            return comparison
            
        except ImportError:
            logger.warning("persim не установлен, используем упрощенное сравнение")
            
            # Упрощенное сравнение без persim
            comparison = {}
            
            for dim in set(diagrams1.keys()) | set(diagrams2.keys()):
                dgm1 = diagrams1.get(dim, np.array([]).reshape(0, 2))
                dgm2 = diagrams2.get(dim, np.array([]).reshape(0, 2))
                
                # Простое сравнение количества признаков
                comparison[f'feature_count_diff_dim_{dim}'] = abs(len(dgm1) - len(dgm2))
            
            return comparison


def analyze_molecular_persistence(coordinates: torch.Tensor,
                                atomic_numbers: Optional[torch.Tensor] = None,
                                method: str = 'gudhi',
                                max_dimension: int = 2) -> Dict[str, any]:
    """
    Удобная функция для полного анализа персистентной гомологии молекулы.
    
    Args:
        coordinates: Координаты атомов [N, 3]
        atomic_numbers: Атомные номера [N] (опционально)
        method: Метод вычисления ('gudhi' или 'ripser')
        max_dimension: Максимальная размерность
    
    Returns:
        Dict: Полные результаты анализа
    """
    analyzer = PersistentHomologyAnalyzer(max_dimension=max_dimension, method=method)
    
    # Вычисляем персистентную гомологию
    persistence_result = analyzer.compute_persistence_from_coordinates(coordinates)
    
    # Добавляем ландшафты для векторизации
    landscapes = analyzer.compute_persistence_landscapes(persistence_result['diagrams'])
    persistence_result['landscapes'] = landscapes
    
    return persistence_result


def batch_persistence_analysis(coordinates_list: List[torch.Tensor],
                             atomic_numbers_list: Optional[List[torch.Tensor]] = None,
                             method: str = 'gudhi',
                             max_dimension: int = 2) -> List[Dict[str, any]]:
    """
    Пакетный анализ персистентной гомологии для множества молекул.
    
    Args:
        coordinates_list: Список координат молекул
        atomic_numbers_list: Список атомных номеров (опционально)
        method: Метод вычисления
        max_dimension: Максимальная размерность
    
    Returns:
        List[Dict]: Результаты анализа для каждой молекулы
    """
    results = []
    
    logger.info(f"Начинаем пакетный анализ персистентности для {len(coordinates_list)} молекул")
    
    for i, coords in enumerate(coordinates_list):
        atomic_nums = atomic_numbers_list[i] if atomic_numbers_list else None
        
        try:
            result = analyze_molecular_persistence(
                coords, atomic_nums, method, max_dimension
            )
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Обработано {i + 1}/{len(coordinates_list)} молекул")
                
        except Exception as e:
            logger.error(f"Ошибка при анализе молекулы {i}: {e}")
            results.append({'error': str(e)})
    
    logger.info(f"Завершен пакетный анализ персистентности")
    
    return results