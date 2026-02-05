"""
Адаптеры для моделей, чтобы они работали с ModelTrainer.

Преобразуют данные из формата ModelTrainer в формат, ожидаемый моделями.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class EGNNAdapter(nn.Module):
    """
    Адаптер для EGNN модели.
    
    Преобразует данные из формата (X, coords) в формат (x, pos, edge_index, batch).
    """
    
    def __init__(self, egnn_model: nn.Module):
        """
        Инициализация адаптера.
        
        Args:
            egnn_model: EGNN модель для адаптации
        """
        super(EGNNAdapter, self).__init__()
        self.egnn_model = egnn_model
        
    def _create_edge_index(self, batch_size: int, max_atoms: int) -> torch.Tensor:
        """
        Создает edge_index для полносвязного графа.
        
        Args:
            batch_size: Размер батча
            max_atoms: Максимальное количество атомов
        
        Returns:
            torch.Tensor: Edge index [2, E]
        """
        edges = []
        for batch_idx in range(batch_size):
            offset = batch_idx * max_atoms
            for i in range(max_atoms):
                for j in range(max_atoms):
                    if i != j:  # Не включаем self-loops
                        edges.append([offset + i, offset + j])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            # Fallback для случая с одним атомом
            return torch.tensor([[0, 0], [0, 0]], dtype=torch.long).t().contiguous()
    
    def _create_batch_tensor(self, batch_size: int, max_atoms: int) -> torch.Tensor:
        """
        Создает batch tensor.
        
        Args:
            batch_size: Размер батча
            max_atoms: Максимальное количество атомов
        
        Returns:
            torch.Tensor: Batch tensor [N]
        """
        batch = []
        for batch_idx in range(batch_size):
            batch.extend([batch_idx] * max_atoms)
        
        return torch.tensor(batch, dtype=torch.long)
    
    def forward(self, X: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через адаптер.
        
        Args:
            X: Node features [batch_size, max_atoms, features]
            coords: Coordinates [batch_size, max_atoms, 3]
        
        Returns:
            torch.Tensor: Predictions [batch_size, 1]
        """
        batch_size, max_atoms, n_features = X.shape
        
        # Преобразуем в формат PyTorch Geometric
        x = X.view(-1, n_features)  # [batch_size * max_atoms, features]
        pos = coords.view(-1, 3)    # [batch_size * max_atoms, 3]
        
        # Создаем edge_index и batch
        edge_index = self._create_edge_index(batch_size, max_atoms)
        batch = self._create_batch_tensor(batch_size, max_atoms)
        
        # Перемещаем на то же устройство
        edge_index = edge_index.to(X.device)
        batch = batch.to(X.device)
        
        # Вызываем EGNN модель
        result = self.egnn_model(x, pos, edge_index, batch)
        
        # Возвращаем предсказания
        if isinstance(result, dict):
            return result['prediction']
        else:
            return result


class SimpleFCNN(nn.Module):
    """
    Простая FCNN для агрегированных молекулярных признаков.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 4, dropout: float = 0.2):
        """
        Инициализация простой FCNN.
        
        Args:
            input_dim: Размерность входных признаков
            hidden_dim: Размерность скрытых слоев
            num_layers: Количество слоев
            dropout: Вероятность dropout
        """
        super(SimpleFCNN, self).__init__()
        
        layers = []
        
        # Входной слой
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # Скрытые слои
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # Выходной слой
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход.
        
        Args:
            x: Входные признаки [batch_size, input_dim]
        
        Returns:
            torch.Tensor: Предсказания [batch_size, 1]
        """
        return self.network(x)


class FCNNAdapter(nn.Module):
    """
    Адаптер для FCNN модели.
    
    Использует простую FCNN для агрегированных признаков молекул.
    """
    
    def __init__(self, input_dim: int):
        """
        Инициализация адаптера.
        
        Args:
            input_dim: Размерность входных признаков
        """
        super(FCNNAdapter, self).__init__()
        self.fcnn_model = SimpleFCNN(
            input_dim=input_dim,
            hidden_dim=128,
            num_layers=4,
            dropout=0.2
        )
        
    def forward(self, X: torch.Tensor, coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Прямой проход через адаптер.
        
        Args:
            X: Features [batch_size, features] (агрегированные признаки молекул)
            coords: Coordinates (не используются для FCNN)
        
        Returns:
            torch.Tensor: Predictions [batch_size, 1]
        """
        return self.fcnn_model(X)


class GCNAdapter(nn.Module):
    """
    Адаптер для GCN модели.
    
    Преобразует данные из формата (X, coords) в формат (x, pos, edge_index, batch).
    """
    
    def __init__(self, gcn_model: nn.Module):
        """
        Инициализация адаптера.
        
        Args:
            gcn_model: GCN модель для адаптации
        """
        super(GCNAdapter, self).__init__()
        self.gcn_model = gcn_model
        
    def _create_edge_index(self, batch_size: int, max_atoms: int) -> torch.Tensor:
        """Создает edge_index для полносвязного графа."""
        edges = []
        for batch_idx in range(batch_size):
            offset = batch_idx * max_atoms
            for i in range(max_atoms):
                for j in range(max_atoms):
                    if i != j:
                        edges.append([offset + i, offset + j])
        
        if edges:
            return torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            return torch.tensor([[0, 0], [0, 0]], dtype=torch.long).t().contiguous()
    
    def _create_batch_tensor(self, batch_size: int, max_atoms: int) -> torch.Tensor:
        """Создает batch tensor."""
        batch = []
        for batch_idx in range(batch_size):
            batch.extend([batch_idx] * max_atoms)
        
        return torch.tensor(batch, dtype=torch.long)
    
    def forward(self, X: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        Прямой проход через адаптер.
        
        Args:
            X: Node features [batch_size, max_atoms, features]
            coords: Coordinates [batch_size, max_atoms, 3]
        
        Returns:
            torch.Tensor: Predictions [batch_size, 1]
        """
        batch_size, max_atoms, n_features = X.shape
        
        # Преобразуем в формат PyTorch Geometric
        x = X.view(-1, n_features)  # [batch_size * max_atoms, features]
        pos = coords.view(-1, 3)    # [batch_size * max_atoms, 3]
        
        # Создаем edge_index и batch
        edge_index = self._create_edge_index(batch_size, max_atoms)
        batch = self._create_batch_tensor(batch_size, max_atoms)
        
        # Перемещаем на то же устройство
        edge_index = edge_index.to(X.device)
        batch = batch.to(X.device)
        
        # Вызываем GCN модель
        result = self.gcn_model(x, pos, edge_index, batch)
        
        # Возвращаем предсказания
        if isinstance(result, dict):
            return result['prediction']
        else:
            return result


def create_model_adapter(model: nn.Module, model_type: str, input_dim: Optional[int] = None) -> nn.Module:
    """
    Создает адаптер для модели.
    
    Args:
        model: Модель для адаптации
        model_type: Тип модели ('egnn', 'fcnn', 'gcn')
        input_dim: Размерность входных данных (для fcnn)
    
    Returns:
        nn.Module: Адаптированная модель
    """
    if model_type == 'egnn':
        return EGNNAdapter(model)
    elif model_type == 'fcnn':
        if input_dim is None:
            raise ValueError("input_dim требуется для FCNN адаптера")
        return FCNNAdapter(input_dim)
    elif model_type == 'gcn':
        return GCNAdapter(model)
    else:
        raise ValueError(f"Неизвестный тип модели для адаптации: {model_type}")