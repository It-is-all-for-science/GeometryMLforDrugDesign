"""
Утилиты для работы с геометрическими моделями.

Содержит функции для тестирования эквивариантности, геометрических трансформаций
и других полезных операций для работы с EGNN и baseline моделями.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.transform import Rotation
import logging

logger = logging.getLogger(__name__)


class GeometricTransforms:
    """
    Класс для применения геометрических трансформаций к молекулярным данным.
    
    Включает вращения, трансляции, отражения и другие трансформации
    для тестирования эквивариантности моделей.
    """
    
    @staticmethod
    def random_rotation_matrix(device: torch.device = torch.device('cpu')) -> torch.Tensor:
        """
        Генерирует случайную матрицу вращения в 3D.
        
        Args:
            device: Устройство для размещения тензора
        
        Returns:
            torch.Tensor: Матрица вращения [3, 3]
        """
        # Используем scipy для генерации случайного вращения
        rotation = Rotation.random()
        rotation_matrix = torch.tensor(rotation.as_matrix(), dtype=torch.float32, device=device)
        
        return rotation_matrix
    
    @staticmethod
    def rotation_matrix_from_axis_angle(axis: torch.Tensor, 
                                      angle: torch.Tensor) -> torch.Tensor:
        """
        Создает матрицу вращения из оси и угла (формула Родрига).
        
        Args:
            axis: Ось вращения [3] (нормализованная)
            angle: Угол вращения в радианах
        
        Returns:
            torch.Tensor: Матрица вращения [3, 3]
        """
        # Нормализуем ось
        axis = F.normalize(axis, dim=0)
        
        # Формула Родрига
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        # Кососимметричная матрица
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ], device=axis.device, dtype=axis.dtype)
        
        # Матрица вращения
        R = (torch.eye(3, device=axis.device, dtype=axis.dtype) + 
             sin_angle * K + 
             (1 - cos_angle) * torch.matmul(K, K))
        
        return R
    
    @staticmethod
    def apply_rotation(pos: torch.Tensor, 
                      rotation_matrix: torch.Tensor) -> torch.Tensor:
        """
        Применяет вращение к координатам.
        
        Args:
            pos: Координаты [N, 3]
            rotation_matrix: Матрица вращения [3, 3]
        
        Returns:
            torch.Tensor: Повернутые координаты [N, 3]
        """
        return torch.matmul(pos, rotation_matrix.T)
    
    @staticmethod
    def apply_translation(pos: torch.Tensor, 
                         translation: torch.Tensor) -> torch.Tensor:
        """
        Применяет трансляцию к координатам.
        
        Args:
            pos: Координаты [N, 3]
            translation: Вектор трансляции [3]
        
        Returns:
            torch.Tensor: Сдвинутые координаты [N, 3]
        """
        return pos + translation
    
    @staticmethod
    def apply_reflection(pos: torch.Tensor, 
                        normal: torch.Tensor) -> torch.Tensor:
        """
        Применяет отражение относительно плоскости.
        
        Args:
            pos: Координаты [N, 3]
            normal: Нормаль к плоскости [3] (нормализованная)
        
        Returns:
            torch.Tensor: Отраженные координаты [N, 3]
        """
        # Нормализуем нормаль
        normal = F.normalize(normal, dim=0)
        
        # Матрица отражения: I - 2 * n * n^T
        reflection_matrix = (torch.eye(3, device=pos.device, dtype=pos.dtype) - 
                           2 * torch.outer(normal, normal))
        
        return torch.matmul(pos, reflection_matrix.T)
    
    @staticmethod
    def add_noise(pos: torch.Tensor, 
                 noise_std: float = 0.01) -> torch.Tensor:
        """
        Добавляет Гауссов шум к координатам.
        
        Args:
            pos: Координаты [N, 3]
            noise_std: Стандартное отклонение шума
        
        Returns:
            torch.Tensor: Координаты с шумом [N, 3]
        """
        noise = torch.randn_like(pos) * noise_std
        return pos + noise


class EquivarianceTest:
    """
    Класс для тестирования эквивариантности геометрических моделей.
    
    Проводит различные тесты для проверки инвариантности и эквивариантности
    к геометрическим трансформациям.
    """
    
    def __init__(self, tolerance: float = 1e-4):
        """
        Инициализация тестера эквивариантности.
        
        Args:
            tolerance: Допустимая погрешность для сравнений
        """
        self.tolerance = tolerance
        self.transforms = GeometricTransforms()
    
    def test_translation_invariance(self, 
                                  model: torch.nn.Module,
                                  x: torch.Tensor,
                                  pos: torch.Tensor,
                                  edge_index: torch.Tensor,
                                  batch: Optional[torch.Tensor] = None,
                                  num_tests: int = 5) -> Dict[str, Any]:
        """
        Тестирует инвариантность к трансляциям.
        
        Args:
            model: Модель для тестирования
            x: Признаки узлов
            pos: Координаты узлов
            edge_index: Индексы ребер
            batch: Индексы батча
            num_tests: Количество тестов
        
        Returns:
            Dict[str, Any]: Результаты тестирования
        """
        model.eval()
        results = []
        
        with torch.no_grad():
            # Исходное предсказание
            original_output = model(x, pos, edge_index, batch)
            original_pred = original_output['prediction']
            
            for i in range(num_tests):
                # Случайная трансляция
                translation = torch.randn(3, device=pos.device) * 5.0
                translated_pos = self.transforms.apply_translation(pos, translation)
                
                # Предсказание для сдвинутых координат
                translated_output = model(x, translated_pos, edge_index, batch)
                translated_pred = translated_output['prediction']
                
                # Проверяем инвариантность
                is_invariant = torch.allclose(original_pred, translated_pred, atol=self.tolerance)
                max_diff = torch.max(torch.abs(original_pred - translated_pred)).item()
                
                results.append({
                    'test_id': i,
                    'translation': translation.cpu().numpy(),
                    'is_invariant': is_invariant,
                    'max_difference': max_diff
                })
        
        # Общая статистика
        passed_tests = sum(1 for r in results if r['is_invariant'])
        success_rate = passed_tests / num_tests
        
        return {
            'test_type': 'translation_invariance',
            'success_rate': success_rate,
            'passed_tests': passed_tests,
            'total_tests': num_tests,
            'individual_results': results,
            'tolerance': self.tolerance
        }
    
    def test_rotation_invariance(self, 
                                model: torch.nn.Module,
                                x: torch.Tensor,
                                pos: torch.Tensor,
                                edge_index: torch.Tensor,
                                batch: Optional[torch.Tensor] = None,
                                num_tests: int = 5) -> Dict[str, Any]:
        """
        Тестирует инвариантность к вращениям.
        
        Args:
            model: Модель для тестирования
            x: Признаки узлов
            pos: Координаты узлов
            edge_index: Индексы ребер
            batch: Индексы батча
            num_tests: Количество тестов
        
        Returns:
            Dict[str, Any]: Результаты тестирования
        """
        model.eval()
        results = []
        
        with torch.no_grad():
            # Исходное предсказание
            original_output = model(x, pos, edge_index, batch)
            original_pred = original_output['prediction']
            
            for i in range(num_tests):
                # Случайное вращение
                rotation_matrix = self.transforms.random_rotation_matrix(pos.device)
                rotated_pos = self.transforms.apply_rotation(pos, rotation_matrix)
                
                # Предсказание для повернутых координат
                rotated_output = model(x, rotated_pos, edge_index, batch)
                rotated_pred = rotated_output['prediction']
                
                # Проверяем инвариантность
                is_invariant = torch.allclose(original_pred, rotated_pred, atol=self.tolerance)
                max_diff = torch.max(torch.abs(original_pred - rotated_pred)).item()
                
                results.append({
                    'test_id': i,
                    'rotation_matrix': rotation_matrix.cpu().numpy(),
                    'is_invariant': is_invariant,
                    'max_difference': max_diff
                })
        
        # Общая статистика
        passed_tests = sum(1 for r in results if r['is_invariant'])
        success_rate = passed_tests / num_tests
        
        return {
            'test_type': 'rotation_invariance',
            'success_rate': success_rate,
            'passed_tests': passed_tests,
            'total_tests': num_tests,
            'individual_results': results,
            'tolerance': self.tolerance
        }
    
    def test_coordinate_equivariance(self, 
                                   model: torch.nn.Module,
                                   x: torch.Tensor,
                                   pos: torch.Tensor,
                                   edge_index: torch.Tensor,
                                   batch: Optional[torch.Tensor] = None,
                                   num_tests: int = 5) -> Dict[str, Any]:
        """
        Тестирует эквивариантность координат к вращениям.
        
        Args:
            model: Модель для тестирования (должна возвращать координаты)
            x: Признаки узлов
            pos: Координаты узлов
            edge_index: Индексы ребер
            batch: Индексы батча
            num_tests: Количество тестов
        
        Returns:
            Dict[str, Any]: Результаты тестирования
        """
        model.eval()
        results = []
        
        with torch.no_grad():
            # Исходное предсказание
            original_output = model(x, pos, edge_index, batch)
            
            # Проверяем, что модель возвращает координаты
            if 'coordinates' not in original_output:
                return {
                    'test_type': 'coordinate_equivariance',
                    'error': 'Модель не возвращает координаты',
                    'success_rate': 0.0
                }
            
            original_coords = original_output['coordinates']
            
            for i in range(num_tests):
                # Случайное вращение
                rotation_matrix = self.transforms.random_rotation_matrix(pos.device)
                rotated_pos = self.transforms.apply_rotation(pos, rotation_matrix)
                
                # Предсказание для повернутых координат
                rotated_output = model(x, rotated_pos, edge_index, batch)
                rotated_coords = rotated_output['coordinates']
                
                # Ожидаемые координаты после вращения
                expected_coords = self.transforms.apply_rotation(original_coords, rotation_matrix)
                
                # Проверяем эквивариантность
                is_equivariant = torch.allclose(rotated_coords, expected_coords, atol=self.tolerance)
                max_diff = torch.max(torch.abs(rotated_coords - expected_coords)).item()
                
                results.append({
                    'test_id': i,
                    'rotation_matrix': rotation_matrix.cpu().numpy(),
                    'is_equivariant': is_equivariant,
                    'max_difference': max_diff
                })
        
        # Общая статистика
        passed_tests = sum(1 for r in results if r['is_equivariant'])
        success_rate = passed_tests / num_tests
        
        return {
            'test_type': 'coordinate_equivariance',
            'success_rate': success_rate,
            'passed_tests': passed_tests,
            'total_tests': num_tests,
            'individual_results': results,
            'tolerance': self.tolerance
        }
    
    def run_full_test_suite(self, 
                           model: torch.nn.Module,
                           x: torch.Tensor,
                           pos: torch.Tensor,
                           edge_index: torch.Tensor,
                           batch: Optional[torch.Tensor] = None,
                           num_tests: int = 5) -> Dict[str, Any]:
        """
        Запускает полный набор тестов эквивариантности.
        
        Args:
            model: Модель для тестирования
            x: Признаки узлов
            pos: Координаты узлов
            edge_index: Индексы ребер
            batch: Индексы батча
            num_tests: Количество тестов для каждого типа
        
        Returns:
            Dict[str, Any]: Полные результаты тестирования
        """
        logger.info(f"Запуск полного набора тестов эквивариантности для {model.__class__.__name__}")
        
        # Тест трансляционной инвариантности
        translation_results = self.test_translation_invariance(
            model, x, pos, edge_index, batch, num_tests
        )
        
        # Тест вращательной инвариантности
        rotation_results = self.test_rotation_invariance(
            model, x, pos, edge_index, batch, num_tests
        )
        
        # Тест эквивариантности координат (если поддерживается)
        coordinate_results = self.test_coordinate_equivariance(
            model, x, pos, edge_index, batch, num_tests
        )
        
        # Общая статистика
        all_tests = [translation_results, rotation_results]
        if 'error' not in coordinate_results:
            all_tests.append(coordinate_results)
        
        overall_success_rate = np.mean([test['success_rate'] for test in all_tests])
        
        results = {
            'model_name': model.__class__.__name__,
            'overall_success_rate': overall_success_rate,
            'translation_invariance': translation_results,
            'rotation_invariance': rotation_results,
            'coordinate_equivariance': coordinate_results,
            'summary': {
                'all_tests_passed': overall_success_rate == 1.0,
                'tolerance': self.tolerance,
                'num_tests_per_type': num_tests
            }
        }
        
        logger.info(f"Тесты завершены. Общий успех: {overall_success_rate:.2%}")
        
        return results


class ModelUtils:
    """
    Утилиты для работы с моделями.
    
    Включает функции для инициализации весов, подсчета параметров,
    визуализации архитектуры и других полезных операций.
    """
    
    @staticmethod
    def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
        """
        Подсчитывает количество параметров в модели.
        
        Args:
            model: PyTorch модель
        
        Returns:
            Dict[str, int]: Статистика параметров
        """
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
    
    @staticmethod
    def initialize_weights(model: torch.nn.Module, method: str = 'xavier_uniform'):
        """
        Инициализирует веса модели.
        
        Args:
            model: PyTorch модель
            method: Метод инициализации ('xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal')
        """
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                if method == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(module.weight)
                elif method == 'xavier_normal':
                    torch.nn.init.xavier_normal_(module.weight)
                elif method == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                elif method == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
    
    @staticmethod
    def get_model_summary(model: torch.nn.Module) -> str:
        """
        Создает текстовое описание архитектуры модели.
        
        Args:
            model: PyTorch модель
        
        Returns:
            str: Описание модели
        """
        param_stats = ModelUtils.count_parameters(model)
        
        summary = f"Модель: {model.__class__.__name__}\n"
        summary += f"Общее количество параметров: {param_stats['total_parameters']:,}\n"
        summary += f"Обучаемые параметры: {param_stats['trainable_parameters']:,}\n"
        summary += f"Необучаемые параметры: {param_stats['non_trainable_parameters']:,}\n\n"
        
        summary += "Архитектура:\n"
        summary += str(model)
        
        return summary
    
    @staticmethod
    def visualize_equivariance_results(test_results: Dict[str, Any], 
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Визуализирует результаты тестов эквивариантности.
        
        Args:
            test_results: Результаты тестов от EquivarianceTest
            save_path: Путь для сохранения графика
        
        Returns:
            plt.Figure: График результатов
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Данные для визуализации
        test_types = ['translation_invariance', 'rotation_invariance', 'coordinate_equivariance']
        test_names = ['Translation\nInvariance', 'Rotation\nInvariance', 'Coordinate\nEquivariance']
        
        for i, (test_type, test_name) in enumerate(zip(test_types, test_names)):
            ax = axes[i]
            
            if test_type in test_results and 'error' not in test_results[test_type]:
                result = test_results[test_type]
                success_rate = result['success_rate']
                
                # Круговая диаграмма
                sizes = [success_rate, 1 - success_rate]
                colors = ['green', 'red']
                labels = ['Passed', 'Failed']
                
                wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, 
                                                 autopct='%1.1f%%', startangle=90)
                
                ax.set_title(f'{test_name}\n({result["passed_tests"]}/{result["total_tests"]} tests)')
                
            else:
                # Тест не поддерживается или произошла ошибка
                ax.text(0.5, 0.5, 'Not\nSupported', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=14)
                ax.set_title(test_name)
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.axis('off')
        
        plt.suptitle(f'Equivariance Test Results: {test_results.get("model_name", "Unknown Model")}', 
                    fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"График сохранен: {save_path}")
        
        return fig
    
    @staticmethod
    def compare_model_predictions(models: Dict[str, torch.nn.Module],
                                x: torch.Tensor,
                                pos: torch.Tensor,
                                edge_index: torch.Tensor,
                                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Сравнивает предсказания нескольких моделей.
        
        Args:
            models: Словарь моделей {name: model}
            x: Признаки узлов
            pos: Координаты узлов
            edge_index: Индексы ребер
            batch: Индексы батча
        
        Returns:
            Dict[str, torch.Tensor]: Предсказания каждой модели
        """
        predictions = {}
        
        for name, model in models.items():
            model.eval()
            with torch.no_grad():
                try:
                    output = model(x, pos, edge_index, batch)
                    predictions[name] = output['prediction']
                except Exception as e:
                    logger.error(f"Ошибка при предсказании модели {name}: {e}")
                    predictions[name] = None
        
        return predictions