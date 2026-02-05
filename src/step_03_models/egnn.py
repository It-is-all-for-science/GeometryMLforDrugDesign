"""
Реализация E(n) Equivariant Graph Neural Networks (EGNN).

Содержит эквивариантные слои и модели для молекулярного машинного обучения
с сохранением геометрических симметрий.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple, Dict, Any
import numpy as np
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class EGNNConfig:
    """Конфигурация для EGNN модели."""
    
    # Архитектура
    hidden_dim: int = 128
    num_layers: int = 4
    output_dim: int = 1
    
    # Признаки узлов
    node_feature_dim: int = 11  # Атомные признаки (атомный номер, заряд, etc.)
    edge_feature_dim: int = 0   # Признаки ребер (опционально)
    
    # EGNN параметры
    attention: bool = True
    normalize: bool = True
    tanh: bool = True
    
    # Обучение
    dropout: float = 0.1
    activation: str = 'swish'  # 'relu', 'gelu', 'swish'
    
    # Координаты
    update_coords: bool = False  # Отключаем обновление координат для стабильности
    coord_weight: float = 1.0
    
    def __post_init__(self):
        """Валидация конфигурации."""
        assert self.hidden_dim > 0, "hidden_dim должен быть положительным"
        assert self.num_layers > 0, "num_layers должен быть положительным"
        assert 0 <= self.dropout <= 1, "dropout должен быть в [0, 1]"
        assert self.activation in ['relu', 'gelu', 'swish'], f"Неподдерживаемая активация: {self.activation}"


class EGNNLayer(MessagePassing):
    """
    Один слой E(n) Equivariant Graph Neural Network.
    
    Реализует эквивариантное обновление узлов и координат с сохранением
    инвариантности к вращениям и трансляциям.
    """
    
    def __init__(self, 
                 hidden_dim: int,
                 edge_feature_dim: int = 0,
                 attention: bool = True,
                 normalize: bool = True,
                 tanh: bool = True,
                 activation: str = 'swish',
                 dropout: float = 0.1,
                 update_coords: bool = False):
        """
        Инициализация EGNN слоя.
        
        Args:
            hidden_dim: Размерность скрытых представлений
            edge_feature_dim: Размерность признаков ребер
            attention: Использовать ли механизм внимания
            normalize: Нормализовать ли координаты
            tanh: Применять ли tanh к координатным обновлениям
            activation: Функция активации
            dropout: Вероятность dropout
            update_coords: Обновлять ли координаты (для стабильности можно отключить)
        """
        super(EGNNLayer, self).__init__(aggr='add')
        
        self.hidden_dim = hidden_dim
        self.edge_feature_dim = edge_feature_dim
        self.attention = attention
        self.normalize = normalize
        self.tanh = tanh
        self.dropout = dropout
        self.update_coords = update_coords
        
        # Выбираем функцию активации
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()  # SiLU = Swish
        else:
            raise ValueError(f"Неподдерживаемая активация: {activation}")
        
        # Размерность входа для edge MLP
        edge_input_dim = hidden_dim * 2 + 1 + edge_feature_dim  # h_i, h_j, ||r_ij||, edge_attr
        
        # Edge MLP для вычисления сообщений
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            self.activation
        )
        
        # Node MLP для обновления узлов
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Coordinate MLP для обновления координат
        coord_input_dim = hidden_dim * 2 + 1 + edge_feature_dim  # h_i, h_j, ||r_ij||, edge_attr
        self.coord_mlp = nn.Sequential(
            nn.Linear(coord_input_dim, hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Attention механизм (опционально)
        if self.attention:
            self.attention_mlp = nn.Sequential(
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        # Нормализация слоев
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.coord_norm = nn.LayerNorm(3) if normalize else nn.Identity()
    
    def forward(self, 
                h: torch.Tensor,
                pos: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Прямой проход EGNN слоя.
        
        Args:
            h: Признаки узлов [N, hidden_dim]
            pos: Координаты узлов [N, 3]
            edge_index: Индексы ребер [2, E]
            edge_attr: Признаки ребер [E, edge_feature_dim] (опционально)
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Обновленные признаки узлов и координаты
        """
        # Сохраняем исходные значения для residual connections
        h_residual = h
        pos_residual = pos
        
        # Message passing для обновления узлов
        h_updated = self.propagate(edge_index, h=h, pos=pos, edge_attr=edge_attr)
        
        # Residual connection для узлов
        h = h_residual + h_updated
        h = self.node_norm(h)
        
        # Обновление координат (только если включено)
        if self.update_coords:
            pos_updated = self.update_coordinates(h, pos, edge_index, edge_attr)
            # Residual connection для координат
            pos = pos_residual + pos_updated
            pos = self.coord_norm(pos)
        else:
            # Координаты остаются неизменными
            pos = pos_residual
        
        return h, pos
    
    def message(self, 
                h_i: torch.Tensor,
                h_j: torch.Tensor,
                pos_i: torch.Tensor,
                pos_j: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Вычисляет сообщения между узлами.
        
        Args:
            h_i: Признаки узлов-получателей [E, hidden_dim]
            h_j: Признаки узлов-отправителей [E, hidden_dim]
            pos_i: Координаты узлов-получателей [E, 3]
            pos_j: Координаты узлов-отправителей [E, 3]
            edge_attr: Признаки ребер [E, edge_feature_dim]
        
        Returns:
            torch.Tensor: Сообщения [E, hidden_dim]
        """
        # Вычисляем расстояния (инвариантный признак)
        rel_pos = pos_i - pos_j  # Относительные позиции
        distances = torch.norm(rel_pos, dim=-1, keepdim=True)  # [E, 1]
        
        # Формируем входной вектор для edge MLP
        edge_input = [h_i, h_j, distances]
        
        if edge_attr is not None:
            edge_input.append(edge_attr)
        
        edge_input = torch.cat(edge_input, dim=-1)
        
        # Вычисляем сообщения
        messages = self.edge_mlp(edge_input)
        
        # Применяем attention (опционально)
        if self.attention:
            attention_weights = self.attention_mlp(messages)
            messages = messages * attention_weights
        
        return messages
    
    def update(self, 
               aggr_out: torch.Tensor,
               h: torch.Tensor) -> torch.Tensor:
        """
        Обновляет признаки узлов на основе агрегированных сообщений.
        
        Args:
            aggr_out: Агрегированные сообщения [N, hidden_dim]
            h: Исходные признаки узлов [N, hidden_dim]
        
        Returns:
            torch.Tensor: Обновленные признаки узлов [N, hidden_dim]
        """
        # Объединяем исходные признаки с агрегированными сообщениями
        node_input = torch.cat([h, aggr_out], dim=-1)
        
        # Применяем node MLP
        h_updated = self.node_mlp(node_input)
        
        return h_updated
    
    def update_coordinates(self,
                          h: torch.Tensor,
                          pos: torch.Tensor,
                          edge_index: torch.Tensor,
                          edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Обновляет координаты эквивариантным образом.
        
        Args:
            h: Признаки узлов [N, hidden_dim]
            pos: Координаты узлов [N, 3]
            edge_index: Индексы ребер [2, E]
            edge_attr: Признаки ребер [E, edge_feature_dim]
        
        Returns:
            torch.Tensor: Обновления координат [N, 3]
        """
        row, col = edge_index
        
        # Относительные позиции (эквивариантные)
        rel_pos = pos[row] - pos[col]  # [E, 3]
        
        # Расстояния (инвариантные)
        distances = torch.norm(rel_pos, dim=-1, keepdim=True)  # [E, 1]
        
        # Нормализованные направления (эквивариантные)
        directions = rel_pos / (distances + 1e-8)  # [E, 3]
        
        # Формируем входной вектор для coordinate MLP
        coord_input = [h[row], h[col], distances]
        
        if edge_attr is not None:
            coord_input.append(edge_attr)
        
        coord_input = torch.cat(coord_input, dim=-1)
        
        # Вычисляем веса для координатных обновлений
        coord_weights = self.coord_mlp(coord_input)  # [E, 1]
        
        # Применяем tanh для стабильности (опционально)
        if self.tanh:
            coord_weights = torch.tanh(coord_weights)
        
        # Эквивариантное обновление координат
        coord_updates = coord_weights * directions  # [E, 3]
        
        # Агрегируем обновления для каждого узла
        coord_updates_aggregated = torch.zeros_like(pos)  # [N, 3]
        coord_updates_aggregated.index_add_(0, row, coord_updates)
        
        return coord_updates_aggregated


class EGNNModel(nn.Module):
    """
    Полная EGNN модель для молекулярного машинного обучения.
    
    Состоит из нескольких EGNN слоев с финальным предсказательным слоем.
    """
    
    def __init__(self, config: EGNNConfig):
        """
        Инициализация EGNN модели.
        
        Args:
            config: Конфигурация модели
        """
        super(EGNNModel, self).__init__()
        
        self.config = config
        
        # Входной слой для признаков узлов
        self.node_embedding = nn.Sequential(
            nn.Linear(config.node_feature_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # EGNN слои
        self.egnn_layers = nn.ModuleList([
            EGNNLayer(
                hidden_dim=config.hidden_dim,
                edge_feature_dim=config.edge_feature_dim,
                attention=config.attention,
                normalize=config.normalize,
                tanh=config.tanh,
                activation=config.activation,
                dropout=config.dropout,
                update_coords=config.update_coords
            )
            for _ in range(config.num_layers)
        ])
        
        # Выходной слой
        self.output_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        )
        
        # Глобальная агрегация (для предсказания свойств молекул)
        self.global_pool = 'mean'  # 'mean', 'sum', 'max'
        
        logger.info(f"Инициализирована EGNN модель: {config}")
    
    def forward(self, 
                x: torch.Tensor,
                pos: torch.Tensor,
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                edge_attr: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Прямой проход EGNN модели.
        
        Args:
            x: Признаки узлов [N, node_feature_dim]
            pos: Координаты узлов [N, 3]
            edge_index: Индексы ребер [2, E]
            batch: Индексы батча [N] (для батчевой обработки)
            edge_attr: Признаки ребер [E, edge_feature_dim]
        
        Returns:
            Dict[str, torch.Tensor]: Результаты предсказания
        """
        # Эмбеддинг признаков узлов
        h = self.node_embedding(x)
        
        # Центрируем координаты для трансляционной инвариантности
        if batch is not None:
            # Батчевое центрирование
            pos_centered = self._center_coordinates_batch(pos, batch)
        else:
            # Простое центрирование для одной молекулы
            pos_centered = pos - pos.mean(dim=0, keepdim=True)
        
        # Сохраняем исходные координаты
        pos_original = pos_centered.clone()
        
        # Проходим через EGNN слои
        for layer in self.egnn_layers:
            h, pos_centered = layer(h, pos_centered, edge_index, edge_attr)
        
        # Глобальная агрегация для предсказания свойств молекул
        if batch is not None:
            # Батчевая агрегация
            if self.global_pool == 'mean':
                h_global = self._global_mean_pool(h, batch)
            elif self.global_pool == 'sum':
                h_global = self._global_sum_pool(h, batch)
            elif self.global_pool == 'max':
                h_global = self._global_max_pool(h, batch)
            else:
                raise ValueError(f"Неподдерживаемый тип пулинга: {self.global_pool}")
        else:
            # Простая агрегация для одной молекулы
            if self.global_pool == 'mean':
                h_global = h.mean(dim=0, keepdim=True)
            elif self.global_pool == 'sum':
                h_global = h.sum(dim=0, keepdim=True)
            elif self.global_pool == 'max':
                h_global = h.max(dim=0, keepdim=True)[0]
        
        # Финальное предсказание
        output = self.output_mlp(h_global)
        
        # Возвращаем результаты
        results = {
            'prediction': output,
            'node_features': h,
            'coordinates': pos_centered,
            'coordinate_updates': pos_centered - pos_original
        }
        
        return results
    
    def _center_coordinates_batch(self, pos: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Центрирует координаты для каждой молекулы в батче."""
        pos_centered = pos.clone()
        batch_size = int(batch.max().item()) + 1
        
        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                center = pos[mask].mean(dim=0, keepdim=True)
                pos_centered[mask] = pos[mask] - center
        
        return pos_centered
    
    def _global_mean_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Глобальный mean pooling для батчей."""
        # Альтернативная реализация без torch_scatter
        batch_size = int(batch.max().item()) + 1
        pooled = torch.zeros(batch_size, x.size(1), device=x.device, dtype=x.dtype)
        
        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                pooled[i] = x[mask].mean(dim=0)
        
        return pooled
    
    def _global_sum_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Глобальный sum pooling для батчей."""
        batch_size = int(batch.max().item()) + 1
        pooled = torch.zeros(batch_size, x.size(1), device=x.device, dtype=x.dtype)
        
        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                pooled[i] = x[mask].sum(dim=0)
        
        return pooled
    
    def _global_max_pool(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Глобальный max pooling для батчей."""
        batch_size = int(batch.max().item()) + 1
        pooled = torch.zeros(batch_size, x.size(1), device=x.device, dtype=x.dtype)
        
        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                pooled[i] = x[mask].max(dim=0)[0]
        
        return pooled
    
    def get_num_parameters(self) -> int:
        """Возвращает количество параметров модели."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Возвращает информацию о модели."""
        return {
            'config': self.config,
            'num_parameters': self.get_num_parameters(),
            'num_layers': len(self.egnn_layers),
            'hidden_dim': self.config.hidden_dim,
            'output_dim': self.config.output_dim
        }


def create_egnn_model(node_feature_dim: int = 11,
                     hidden_dim: int = 128,
                     num_layers: int = 4,
                     output_dim: int = 1,
                     **kwargs) -> EGNNModel:
    """
    Удобная функция для создания EGNN модели.
    
    Args:
        node_feature_dim: Размерность признаков узлов
        hidden_dim: Размерность скрытых слоев
        num_layers: Количество EGNN слоев
        output_dim: Размерность выхода
        **kwargs: Дополнительные параметры конфигурации
    
    Returns:
        EGNNModel: Инициализированная модель
    """
    config = EGNNConfig(
        node_feature_dim=node_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        **kwargs
    )
    
    return EGNNModel(config)


def test_equivariance(model: EGNNModel,
                     x: torch.Tensor,
                     pos: torch.Tensor,
                     edge_index: torch.Tensor,
                     batch: Optional[torch.Tensor] = None,
                     tolerance: float = 1e-4) -> Dict[str, bool]:
    """
    Тестирует эквивариантность EGNN модели к вращениям и трансляциям.
    
    Args:
        model: EGNN модель для тестирования
        x: Признаки узлов
        pos: Координаты узлов
        edge_index: Индексы ребер
        batch: Индексы батча
        tolerance: Допустимая погрешность
    
    Returns:
        Dict[str, bool]: Результаты тестов эквивариантности
    """
    model.eval()
    
    with torch.no_grad():
        # Исходное предсказание
        original_output = model(x, pos, edge_index, batch)
        
        # Тест трансляционной инвариантности
        translation = torch.randn(1, 3)
        translated_pos = pos + translation
        translated_output = model(x, translated_pos, edge_index, batch)
        
        translation_invariant = torch.allclose(
            original_output['prediction'],
            translated_output['prediction'],
            atol=tolerance
        )
        
        # Тест вращательной инвариантности
        # Генерируем случайную матрицу вращения
        angle = torch.rand(1) * 2 * np.pi
        axis = F.normalize(torch.randn(3), dim=0)
        
        # Матрица вращения вокруг произвольной оси
        cos_angle = torch.cos(angle)
        sin_angle = torch.sin(angle)
        
        K = torch.tensor([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        
        rotation_matrix = (torch.eye(3) + 
                          sin_angle * K + 
                          (1 - cos_angle) * torch.matmul(K, K))
        
        rotated_pos = torch.matmul(pos, rotation_matrix.T)
        rotated_output = model(x, rotated_pos, edge_index, batch)
        
        rotation_invariant = torch.allclose(
            original_output['prediction'],
            rotated_output['prediction'],
            atol=tolerance
        )
        
        # Тест эквивариантности координат
        expected_rotated_coords = torch.matmul(
            original_output['coordinates'], 
            rotation_matrix.T
        )
        
        coordinate_equivariant = torch.allclose(
            rotated_output['coordinates'],
            expected_rotated_coords,
            atol=tolerance
        )
    
    results = {
        'translation_invariant': translation_invariant,
        'rotation_invariant': rotation_invariant,
        'coordinate_equivariant': coordinate_equivariant,
        'all_tests_passed': translation_invariant and rotation_invariant and coordinate_equivariant
    }
    
    logger.info(f"Тесты эквивариантности: {results}")
    
    return results