"""
Baseline модели для сравнения с EGNN.

Содержит простые нейронные сети без геометрических prior-ов
и табличные методы для демонстрации важности inductive biases.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, GraphSAGE, 
    global_mean_pool, global_max_pool, global_add_pool,
    Set2Set, SAGPooling, BatchNorm
)
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class BaselineConfig:
    """Конфигурация для baseline моделей."""
    
    # Архитектура
    hidden_dim: int = 128
    num_layers: int = 4
    output_dim: int = 1
    
    # Признаки
    node_feature_dim: int = 11
    use_coordinates: bool = True  # Использовать ли координаты как признаки
    
    # Обучение
    dropout: float = 0.1
    activation: str = 'swish'  # Изменили на более современную активацию
    
    # Пулинг
    pooling: str = 'attention'  # 'mean', 'max', 'add', 'set2set', 'attention'
    
    # Улучшения архитектуры
    use_batch_norm: bool = True
    use_residual: bool = True
    use_attention: bool = True
    
    # Специфичные параметры GCN
    gcn_type: str = 'gat'  # 'gcn', 'gat', 'gin', 'sage'
    num_heads: int = 4  # Для GAT
    
    def __post_init__(self):
        """Валидация конфигурации."""
        assert self.hidden_dim > 0, "hidden_dim должен быть положительным"
        assert self.num_layers > 0, "num_layers должен быть положительным"
        assert 0 <= self.dropout <= 1, "dropout должен быть в [0, 1]"
        assert self.activation in ['relu', 'gelu', 'swish'], f"Неподдерживаемая активация: {self.activation}"
        assert self.pooling in ['mean', 'max', 'add', 'set2set', 'attention'], f"Неподдерживаемый пулинг: {self.pooling}"
        assert self.gcn_type in ['gcn', 'gat', 'gin', 'sage'], f"Неподдерживаемый тип GCN: {self.gcn_type}"


class FCNNBaseline(nn.Module):
    """
    Улучшенная baseline модель: полносвязная нейронная сеть с современными техниками.
    
    Использует batch normalization, residual connections, attention механизм
    для атомов, но все еще без геометрических prior-ов.
    """
    
    def __init__(self, config: BaselineConfig):
        """
        Инициализация улучшенной FCNN baseline модели.
        
        Args:
            config: Конфигурация модели
        """
        super(FCNNBaseline, self).__init__()
        
        self.config = config
        
        # Вычисляем размерность входа
        input_dim = config.node_feature_dim
        if config.use_coordinates:
            input_dim += 3  # x, y, z координаты
        
        # Выбираем функцию активации
        if config.activation == 'relu':
            self.activation = nn.ReLU()
        elif config.activation == 'gelu':
            self.activation = nn.GELU()
        elif config.activation == 'swish':
            self.activation = nn.SiLU()
        
        # Входной слой
        self.input_projection = nn.Linear(input_dim, config.hidden_dim)
        self.input_norm = nn.BatchNorm1d(config.hidden_dim) if config.use_batch_norm else nn.Identity()
        
        # Остаточные блоки
        self.residual_blocks = nn.ModuleList()
        for i in range(config.num_layers):
            block = ResidualBlock(
                config.hidden_dim, 
                config.dropout, 
                self.activation,
                use_batch_norm=config.use_batch_norm
            )
            self.residual_blocks.append(block)
        
        # Механизм внимания для атомов (если включен)
        if config.use_attention:
            self.atom_attention = AtomAttention(config.hidden_dim)
        
        # Выходной слой
        self.output_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            self.activation,
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
        # Выбираем функцию пулинга
        if config.pooling == 'mean':
            self.pool = global_mean_pool
        elif config.pooling == 'max':
            self.pool = global_max_pool
        elif config.pooling == 'add':
            self.pool = global_add_pool
        elif config.pooling == 'set2set':
            self.pool = Set2Set(config.hidden_dim, processing_steps=3)
            # Обновляем размерность для Set2Set
            self.output_mlp = nn.Sequential(
                nn.Linear(2 * config.hidden_dim, config.hidden_dim),
                self.activation,
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.output_dim)
            )
        elif config.pooling == 'attention':
            self.pool = AttentionPooling(config.hidden_dim)
        
        logger.info(f"Инициализирована улучшенная FCNN baseline модель: {config}")
    
    def forward(self, 
                x: torch.Tensor,
                pos: torch.Tensor,
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Прямой проход улучшенной FCNN baseline модели.
        
        Args:
            x: Признаки узлов [N, node_feature_dim]
            pos: Координаты узлов [N, 3]
            edge_index: Индексы ребер [2, E] (не используется)
            batch: Индексы батча [N]
            **kwargs: Дополнительные аргументы (игнорируются)
        
        Returns:
            Dict[str, torch.Tensor]: Результаты предсказания
        """
        # Формируем входной вектор
        if self.config.use_coordinates:
            node_input = torch.cat([x, pos], dim=-1)
        else:
            node_input = x
        
        # Входная проекция
        h = self.input_projection(node_input)
        h = self.input_norm(h)
        h = self.activation(h)
        
        # Остаточные блоки
        for block in self.residual_blocks:
            h = block(h)
        
        # Внимание к атомам (если включено)
        if self.config.use_attention:
            h = self.atom_attention(h, batch)
        
        # Глобальная агрегация
        if batch is not None:
            if self.config.pooling == 'set2set':
                h_global = self.pool(h, batch)
            elif self.config.pooling == 'attention':
                h_global = self.pool(h, batch)
            else:
                h_global = self.pool(h, batch)
        else:
            # Простая агрегация для одной молекулы
            if self.config.pooling == 'mean':
                h_global = h.mean(dim=0, keepdim=True)
            elif self.config.pooling == 'max':
                h_global = h.max(dim=0, keepdim=True)[0]
            elif self.config.pooling == 'add':
                h_global = h.sum(dim=0, keepdim=True)
            elif self.config.pooling in ['set2set', 'attention']:
                # Для одной молекулы используем mean pooling
                h_global = h.mean(dim=0, keepdim=True)
        
        # Финальное предсказание
        output = self.output_mlp(h_global)
        
        return {
            'prediction': output,
            'node_features': h
        }
    
    def get_num_parameters(self) -> int:
        """Возвращает количество параметров модели."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Residual блок для FCNN."""
    
    def __init__(self, hidden_dim: int, dropout: float, activation: nn.Module, use_batch_norm: bool = True):
        super().__init__()
        
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.norm2 = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        
        out = self.linear1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)
        
        out = self.linear2(out)
        out = self.norm2(out)
        
        out += residual  # Остаточное соединение
        out = self.activation(out)
        
        return out


class AtomAttention(nn.Module):
    """Attention механизм для атомов."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, batch=None):
        if batch is not None:
            # Батчевая обработка - сложнее реализовать
            # Пока просто возвращаем исходные признаки
            return x
        else:
            # Одна молекула
            x_input = x.unsqueeze(0)  # [1, N, hidden_dim]
            attn_out, _ = self.attention(x_input, x_input, x_input)
            attn_out = attn_out.squeeze(0)  # [N, hidden_dim]
            return self.norm(x + attn_out)


class AttentionPooling(nn.Module):
    """Attention-based pooling для графов."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1)
    
    def forward(self, x, batch=None):
        # Вычисляем веса внимания
        weights = self.attention_weights(x)  # [N, 1]
        weights = torch.softmax(weights, dim=0)
        
        if batch is not None:
            # Батчевая обработка
            output = []
            for batch_idx in torch.unique(batch):
                mask = batch == batch_idx
                batch_x = x[mask]
                batch_weights = weights[mask]
                batch_weights = torch.softmax(batch_weights, dim=0)
                pooled = torch.sum(batch_x * batch_weights, dim=0, keepdim=True)
                output.append(pooled)
            return torch.cat(output, dim=0)
        else:
            # Одна молекула
            return torch.sum(x * weights, dim=0, keepdim=True)


class GCNBaseline(nn.Module):
    """
    Улучшенная baseline модель: Graph Convolutional Network с современными техниками.
    
    Использует современные graph layers (GAT, GIN, GraphSAGE), batch normalization,
    residual connections, но все еще без геометрических prior-ов.
    """
    
    def __init__(self, config: BaselineConfig):
        """
        Инициализация улучшенной GCN baseline модели.
        
        Args:
            config: Конфигурация модели
        """
        super(GCNBaseline, self).__init__()
        
        self.config = config
        
        # Входной слой
        input_dim = config.node_feature_dim
        if config.use_coordinates:
            input_dim += 3
        
        self.input_projection = nn.Linear(input_dim, config.hidden_dim)
        self.input_norm = BatchNorm(config.hidden_dim) if config.use_batch_norm else nn.Identity()
        
        # Выбираем функцию активации
        if config.activation == 'relu':
            self.activation = nn.ReLU()
        elif config.activation == 'gelu':
            self.activation = nn.GELU()
        elif config.activation == 'swish':
            self.activation = nn.SiLU()
        
        # Графовые сверточные слои
        self.graph_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        for i in range(config.num_layers):
            # Выбираем тип графового слоя
            if config.gcn_type == 'gcn':
                layer = GCNConv(config.hidden_dim, config.hidden_dim)
            elif config.gcn_type == 'gat':
                layer = GATConv(
                    config.hidden_dim, 
                    config.hidden_dim // config.num_heads,
                    heads=config.num_heads,
                    dropout=config.dropout,
                    concat=True
                )
            elif config.gcn_type == 'gin':
                mlp = nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    self.activation,
                    nn.Linear(config.hidden_dim, config.hidden_dim)
                )
                layer = GINConv(mlp)
            elif config.gcn_type == 'sage':
                layer = GraphSAGE(
                    config.hidden_dim,
                    config.hidden_dim,
                    num_layers=1,
                    dropout=config.dropout
                )
            
            self.graph_layers.append(layer)
            
            # Batch normalization для каждого слоя
            if config.use_batch_norm:
                self.layer_norms.append(BatchNorm(config.hidden_dim))
            else:
                self.layer_norms.append(nn.Identity())
        
        # Выбираем функцию пулинга
        if config.pooling == 'mean':
            self.pool = global_mean_pool
            pool_output_dim = config.hidden_dim
        elif config.pooling == 'max':
            self.pool = global_max_pool
            pool_output_dim = config.hidden_dim
        elif config.pooling == 'add':
            self.pool = global_add_pool
            pool_output_dim = config.hidden_dim
        elif config.pooling == 'set2set':
            self.pool = Set2Set(config.hidden_dim, processing_steps=3)
            pool_output_dim = 2 * config.hidden_dim
        elif config.pooling == 'attention':
            self.pool = AttentionPooling(config.hidden_dim)
            pool_output_dim = config.hidden_dim
        
        # Выходной MLP с остаточными соединениями
        self.output_mlp = nn.Sequential(
            nn.Linear(pool_output_dim, config.hidden_dim),
            self.activation,
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            self.activation,
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        logger.info(f"Инициализирована улучшенная GCN baseline модель: {config}")
    
    def forward(self, 
                x: torch.Tensor,
                pos: torch.Tensor,
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None,
                **kwargs) -> Dict[str, torch.Tensor]:
        """
        Прямой проход улучшенной GCN baseline модели.
        
        Args:
            x: Признаки узлов [N, node_feature_dim]
            pos: Координаты узлов [N, 3]
            edge_index: Индексы ребер [2, E]
            batch: Индексы батча [N]
            **kwargs: Дополнительные аргументы (игнорируются)
        
        Returns:
            Dict[str, torch.Tensor]: Результаты предсказания
        """
        # Формируем входной вектор
        if self.config.use_coordinates:
            node_input = torch.cat([x, pos], dim=-1)
        else:
            node_input = x
        
        # Проекция входа
        h = self.input_projection(node_input)
        h = self.input_norm(h)
        h = self.activation(h)
        h = self.dropout(h)
        
        # Графовые сверточные слои с остаточными соединениями
        for i, (graph_layer, layer_norm) in enumerate(zip(self.graph_layers, self.layer_norms)):
            h_residual = h
            
            # Применяем графовый слой
            if self.config.gcn_type == 'sage':
                # GraphSAGE требует специальной обработки
                h = graph_layer(h, edge_index)[0]  # Берем только выход, не промежуточные состояния
            else:
                h = graph_layer(h, edge_index)
            
            # Пакетная нормализация
            h = layer_norm(h)
            
            # Остаточное соединение (если размерности совпадают)
            if self.config.use_residual and h.shape == h_residual.shape:
                h = h + h_residual
            
            # Активация и dropout
            h = self.activation(h)
            h = self.dropout(h)
        
        # Глобальная агрегация
        if batch is not None:
            if self.config.pooling == 'set2set':
                h_global = self.pool(h, batch)
            elif self.config.pooling == 'attention':
                h_global = self.pool(h, batch)
            else:
                h_global = self.pool(h, batch)
        else:
            # Простая агрегация для одной молекулы
            if self.config.pooling == 'mean':
                h_global = h.mean(dim=0, keepdim=True)
            elif self.config.pooling == 'max':
                h_global = h.max(dim=0, keepdim=True)[0]
            elif self.config.pooling == 'add':
                h_global = h.sum(dim=0, keepdim=True)
            elif self.config.pooling in ['set2set', 'attention']:
                # Для одной молекулы используем mean pooling
                h_global = h.mean(dim=0, keepdim=True)
        
        # Финальное предсказание
        output = self.output_mlp(h_global)
        
        return {
            'prediction': output,
            'node_features': h
        }
    
    def get_num_parameters(self) -> int:
        """Возвращает количество параметров модели."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TabularBaseline:
    """
    Улучшенная baseline модель: табличные методы с расширенными химическими дескрипторами.
    
    Использует традиционные ML алгоритмы (Random Forest, Gradient Boosting, Extra Trees)
    с более сложными ручно созданными признаками для демонстрации важности
    геометрических inductive biases.
    """
    
    def __init__(self, 
                 model_type: str = 'random_forest',
                 **model_kwargs):
        """
        Инициализация улучшенной табличной baseline модели.
        
        Args:
            model_type: Тип модели ('random_forest', 'extra_trees', 'gradient_boosting', 'ridge')
            **model_kwargs: Параметры для модели
        """
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        # Создаем модель с улучшенными параметрами
        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=model_kwargs.get('n_estimators', 200),  # Увеличили количество деревьев
                max_depth=model_kwargs.get('max_depth', 15),  # Ограничили глубину
                min_samples_split=model_kwargs.get('min_samples_split', 5),
                min_samples_leaf=model_kwargs.get('min_samples_leaf', 2),
                max_features=model_kwargs.get('max_features', 'sqrt'),
                random_state=model_kwargs.get('random_state', 42),
                n_jobs=model_kwargs.get('n_jobs', 1)  # Для Windows совместимости
            )
        elif model_type == 'extra_trees':
            self.model = ExtraTreesRegressor(
                n_estimators=model_kwargs.get('n_estimators', 200),
                max_depth=model_kwargs.get('max_depth', 15),
                min_samples_split=model_kwargs.get('min_samples_split', 5),
                min_samples_leaf=model_kwargs.get('min_samples_leaf', 2),
                max_features=model_kwargs.get('max_features', 'sqrt'),
                random_state=model_kwargs.get('random_state', 42),
                n_jobs=model_kwargs.get('n_jobs', 1)
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=model_kwargs.get('n_estimators', 200),
                max_depth=model_kwargs.get('max_depth', 6),
                learning_rate=model_kwargs.get('learning_rate', 0.05),  # Уменьшили learning rate
                subsample=model_kwargs.get('subsample', 0.8),
                min_samples_split=model_kwargs.get('min_samples_split', 5),
                min_samples_leaf=model_kwargs.get('min_samples_leaf', 2),
                random_state=model_kwargs.get('random_state', 42)
            )
        elif model_type == 'ridge':
            self.model = Ridge(
                alpha=model_kwargs.get('alpha', 1.0),
                random_state=model_kwargs.get('random_state', 42)
            )
        else:
            raise ValueError(f"Неподдерживаемый тип модели: {model_type}")
        
        logger.info(f"Инициализирована улучшенная табличная baseline модель: {model_type}")
    
    def extract_features(self, 
                        x: torch.Tensor,
                        pos: torch.Tensor,
                        edge_index: torch.Tensor,
                        batch: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Извлекает расширенные ручные признаки из молекулярных данных.
        
        Args:
            x: Признаки узлов [N, node_feature_dim]
            pos: Координаты узлов [N, 3]
            edge_index: Индексы ребер [2, E]
            batch: Индексы батча [N]
        
        Returns:
            np.ndarray: Матрица признаков [batch_size, num_features]
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if isinstance(pos, torch.Tensor):
            pos = pos.detach().cpu().numpy()
        if isinstance(edge_index, torch.Tensor):
            edge_index = edge_index.detach().cpu().numpy()
        if batch is not None and isinstance(batch, torch.Tensor):
            batch = batch.detach().cpu().numpy()
        
        features_list = []
        
        if batch is not None:
            # Батчевая обработка
            unique_batches = np.unique(batch)
            
            for batch_idx in unique_batches:
                mask = batch == batch_idx
                mol_x = x[mask]
                mol_pos = pos[mask]
                
                # Извлекаем edge_index для данной молекулы
                mol_edge_mask = np.isin(edge_index[0], np.where(mask)[0]) & np.isin(edge_index[1], np.where(mask)[0])
                mol_edge_index = edge_index[:, mol_edge_mask]
                
                # Переиндексируем edges для локальной молекулы
                if mol_edge_index.shape[1] > 0:
                    node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(np.where(mask)[0])}
                    mol_edge_index_local = np.array([[node_mapping[int(edge_index[0, i])], node_mapping[int(edge_index[1, i])]] 
                                                   for i in range(mol_edge_index.shape[1]) 
                                                   if int(edge_index[0, i]) in node_mapping and int(edge_index[1, i]) in node_mapping]).T
                    
                    if mol_edge_index_local.size == 0:
                        mol_edge_index_local = None
                else:
                    mol_edge_index_local = None
                
                # Извлекаем признаки для одной молекулы
                mol_features = self._extract_single_molecule_features(mol_x, mol_pos, mol_edge_index_local)
                features_list.append(mol_features)
        else:
            # Одна молекула
            mol_features = self._extract_single_molecule_features(x, pos, edge_index)
            features_list.append(mol_features)
        
        return np.array(features_list)
    
    def _extract_single_molecule_features(self, 
                                        x: np.ndarray,
                                        pos: np.ndarray,
                                        edge_index: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Извлекает расширенные признаки для одной молекулы.
        
        Args:
            x: Признаки узлов [N, node_feature_dim]
            pos: Координаты узлов [N, 3]
            edge_index: Индексы ребер [2, E]
        
        Returns:
            np.ndarray: Вектор признаков
        """
        features = []
        
        # 1. Базовые атомные статистики
        if len(x) > 0:
            atomic_numbers = x[:, 0]  # Предполагаем, что первый признак - атомный номер
            
            features.extend([
                len(x),  # Количество атомов
                np.mean(atomic_numbers),  # Средний атомный номер
                np.std(atomic_numbers),   # Стд атомного номера
                np.max(atomic_numbers),   # Максимальный атомный номер
                np.min(atomic_numbers),   # Минимальный атомный номер
                np.median(atomic_numbers), # Медианный атомный номер
                len(np.unique(atomic_numbers)),  # Количество уникальных типов атомов
            ])
            
            # Распределение атомных номеров
            for z in [1, 6, 7, 8, 9, 15, 16, 17]:  # H, C, N, O, F, P, S, Cl
                features.append(np.sum(atomic_numbers == z))  # Количество атомов каждого типа
            
            # Дополнительные атомные признаки (если есть)
            if x.shape[1] > 1:
                for i in range(1, min(x.shape[1], 5)):  # Берем до 4 дополнительных признаков
                    features.extend([
                        np.mean(x[:, i]),
                        np.std(x[:, i]),
                        np.max(x[:, i]),
                        np.min(x[:, i])
                    ])
        else:
            features.extend([0.0] * (7 + 8 + 16))  # Заполняем нулями
        
        # 2. Геометрические признаки
        if len(pos) > 1:
            # Расстояния между атомами
            distances = []
            for i in range(len(pos)):
                for j in range(i + 1, len(pos)):
                    dist = np.linalg.norm(pos[i] - pos[j])
                    distances.append(dist)
            
            distances = np.array(distances)
            
            features.extend([
                np.mean(distances),    # Среднее расстояние
                np.std(distances),     # Стд расстояний
                np.min(distances),     # Минимальное расстояние
                np.max(distances),     # Максимальное расстояние
                np.median(distances),  # Медианное расстояние
                np.percentile(distances, 25),  # 25-й перцентиль
                np.percentile(distances, 75),  # 75-й перцентиль
            ])
            
            # Размеры молекулы
            mol_min = np.min(pos, axis=0)
            mol_max = np.max(pos, axis=0)
            mol_size = mol_max - mol_min
            mol_center = (mol_max + mol_min) / 2
            
            features.extend([
                np.prod(mol_size),     # Объем bounding box
                np.sum(mol_size),      # Периметр bounding box
                np.max(mol_size),      # Максимальный размер
                np.min(mol_size),      # Минимальный размер
                np.mean(mol_size),     # Средний размер
                np.std(mol_size),      # Стд размеров
            ])
            
            # Центр масс и радиус гирации
            center_of_mass = np.mean(pos, axis=0)
            radius_of_gyration = np.sqrt(np.mean(np.sum((pos - center_of_mass)**2, axis=1)))
            
            # Моменты инерции (упрощенные)
            inertia_tensor = np.zeros((3, 3))
            for i, atom_pos in enumerate(pos):
                r = atom_pos - center_of_mass
                inertia_tensor += np.outer(r, r)
            
            eigenvalues = np.linalg.eigvals(inertia_tensor)
            eigenvalues = np.sort(eigenvalues)
            
            features.extend([
                radius_of_gyration,
                eigenvalues[0],  # Наименьший момент инерции
                eigenvalues[1],  # Средний момент инерции
                eigenvalues[2],  # Наибольший момент инерции
                eigenvalues[2] / eigenvalues[0] if eigenvalues[0] > 1e-6 else 0,  # Анизотропия
            ])
            
            # Сферичность и другие дескрипторы формы
            asphericity = eigenvalues[2] - 0.5 * (eigenvalues[0] + eigenvalues[1])
            acylindricity = eigenvalues[1] - eigenvalues[0]
            
            features.extend([
                asphericity,
                acylindricity,
                asphericity / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]) if np.sum(eigenvalues) > 1e-6 else 0
            ])
            
        else:
            # Заполняем нулями для молекул с одним атомом
            features.extend([0.0] * (7 + 6 + 5 + 3))
        
        # 3. Топологические признаки (если есть edge_index)
        if edge_index is not None and len(edge_index[0]) > 0:
            # Степени узлов
            degrees = np.bincount(edge_index[0], minlength=len(pos)) + np.bincount(edge_index[1], minlength=len(pos))
            
            features.extend([
                np.mean(degrees),      # Средняя степень
                np.std(degrees),       # Стд степеней
                np.max(degrees),       # Максимальная степень
                np.min(degrees),       # Минимальная степень
                len(edge_index[0]),    # Количество ребер
                len(edge_index[0]) / len(pos) if len(pos) > 0 else 0,  # Плотность графа
            ])
            
            # Распределение степеней
            for degree in range(1, 5):  # Количество узлов со степенью 1, 2, 3, 4
                features.append(np.sum(degrees == degree))
            
        else:
            features.extend([0.0] * (6 + 4))
        
        # 4. Химические дескрипторы (упрощенные)
        if len(x) > 0:
            atomic_numbers = x[:, 0]
            
            # Молекулярная масса (приблизительная)
            atomic_masses = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 15: 30.974, 16: 32.065, 17: 35.453}
            molecular_weight = sum(atomic_masses.get(int(z), 12.0) for z in atomic_numbers)
            
            # Количество тяжелых атомов (не водород)
            heavy_atoms = np.sum(atomic_numbers > 1)
            
            # Соотношение C/N/O
            c_count = np.sum(atomic_numbers == 6)
            n_count = np.sum(atomic_numbers == 7)
            o_count = np.sum(atomic_numbers == 8)
            
            features.extend([
                molecular_weight,
                heavy_atoms,
                c_count / len(atomic_numbers) if len(atomic_numbers) > 0 else 0,  # Доля углерода
                n_count / len(atomic_numbers) if len(atomic_numbers) > 0 else 0,  # Доля азота
                o_count / len(atomic_numbers) if len(atomic_numbers) > 0 else 0,  # Доля кислорода
                (c_count + n_count + o_count) / len(atomic_numbers) if len(atomic_numbers) > 0 else 0,  # Доля CNO
            ])
        else:
            features.extend([0.0] * 6)
        
        return np.array(features)
    
    def fit(self, 
            x: torch.Tensor,
            pos: torch.Tensor,
            edge_index: torch.Tensor,
            y: torch.Tensor,
            batch: Optional[torch.Tensor] = None) -> 'TabularBaseline':
        """
        Обучает табличную модель.
        
        Args:
            x: Признаки узлов
            pos: Координаты узлов
            edge_index: Индексы ребер
            y: Целевые значения
            batch: Индексы батча
        
        Returns:
            TabularBaseline: Обученная модель
        """
        # Извлекаем признаки
        features = self.extract_features(x, pos, edge_index, batch)
        
        # Проверяем на NaN и Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Нормализуем признаки
        features_scaled = self.scaler.fit_transform(features)
        
        # Подготавливаем целевые значения
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        
        # Обучаем модель
        self.model.fit(features_scaled, y.ravel())
        self.is_fitted = True
        
        logger.info(f"Обучена улучшенная табличная модель на {len(features)} образцах с {features.shape[1]} признаками")
        
        return self
    
    def predict(self, 
                x: torch.Tensor,
                pos: torch.Tensor,
                edge_index: torch.Tensor,
                batch: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Делает предсказания с помощью табличной модели.
        
        Args:
            x: Признаки узлов
            pos: Координаты узлов
            edge_index: Индексы ребер
            batch: Индексы батча
        
        Returns:
            Dict[str, torch.Tensor]: Результаты предсказания
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите fit() сначала.")
        
        # Извлекаем признаки
        features = self.extract_features(x, pos, edge_index, batch)
        
        # Проверяем на NaN и Inf
        features = np.nan_to_num(features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Нормализуем признаки
        features_scaled = self.scaler.transform(features)
        
        # Делаем предсказания
        predictions = self.model.predict(features_scaled)
        
        # Конвертируем в torch tensor
        predictions_tensor = torch.tensor(predictions, dtype=torch.float32).unsqueeze(-1)
        
        return {
            'prediction': predictions_tensor,
            'features': torch.tensor(features, dtype=torch.float32)
        }
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Возвращает важность признаков (если поддерживается моделью).
        
        Returns:
            Optional[np.ndarray]: Важность признаков
        """
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None
    
    def get_feature_names(self) -> list:
        """
        Возвращает названия признаков для интерпретации.
        
        Returns:
            list: Список названий признаков
        """
        feature_names = []
        
        # Базовые атомные статистики
        feature_names.extend([
            'num_atoms', 'mean_atomic_num', 'std_atomic_num', 'max_atomic_num', 
            'min_atomic_num', 'median_atomic_num', 'num_unique_atoms'
        ])
        
        # Количество атомов каждого типа
        for element in ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl']:
            feature_names.append(f'num_{element}')
        
        # Дополнительные атомные признаки
        for i in range(1, 5):
            for stat in ['mean', 'std', 'max', 'min']:
                feature_names.append(f'atom_feature_{i}_{stat}')
        
        # Геометрические признаки
        feature_names.extend([
            'mean_distance', 'std_distance', 'min_distance', 'max_distance', 
            'median_distance', 'distance_25p', 'distance_75p',
            'bbox_volume', 'bbox_perimeter', 'bbox_max_size', 'bbox_min_size',
            'bbox_mean_size', 'bbox_std_size',
            'radius_of_gyration', 'inertia_min', 'inertia_mid', 'inertia_max', 'anisotropy',
            'asphericity', 'acylindricity', 'relative_asphericity'
        ])
        
        # Топологические признаки
        feature_names.extend([
            'mean_degree', 'std_degree', 'max_degree', 'min_degree', 
            'num_edges', 'graph_density'
        ])
        
        for degree in range(1, 5):
            feature_names.append(f'nodes_degree_{degree}')
        
        # Химические дескрипторы
        feature_names.extend([
            'molecular_weight', 'heavy_atoms', 'carbon_fraction', 
            'nitrogen_fraction', 'oxygen_fraction', 'cno_fraction'
        ])
        
        return feature_names


def create_baseline_model(model_type: str = 'fcnn',
                         node_feature_dim: int = 11,
                         hidden_dim: int = 128,
                         num_layers: int = 4,
                         output_dim: int = 1,
                         **kwargs) -> Any:
    """
    Удобная функция для создания улучшенной baseline модели.
    
    Args:
        model_type: Тип модели ('fcnn', 'gcn', 'random_forest', 'extra_trees', 'gradient_boosting', 'ridge')
        node_feature_dim: Размерность признаков узлов
        hidden_dim: Размерность скрытых слоев
        num_layers: Количество слоев
        output_dim: Размерность выхода
        **kwargs: Дополнительные параметры
    
    Returns:
        Any: Инициализированная baseline модель
    """
    if model_type in ['fcnn', 'gcn']:
        config = BaselineConfig(
            node_feature_dim=node_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            output_dim=output_dim,
            **kwargs
        )
        
        if model_type == 'fcnn':
            return FCNNBaseline(config)
        elif model_type == 'gcn':
            return GCNBaseline(config)
    
    elif model_type in ['random_forest', 'extra_trees', 'gradient_boosting', 'ridge']:
        return TabularBaseline(model_type=model_type, **kwargs)
    
    else:
        raise ValueError(f"Неподдерживаемый тип модели: {model_type}. "
                        f"Доступные типы: fcnn, gcn, random_forest, extra_trees, gradient_boosting, ridge")