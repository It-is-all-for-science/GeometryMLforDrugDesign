"""
EGNN модель, адаптированная для белок-лигандных взаимодействий.

Расширяет базовую EGNN архитектуру для работы с гетерогенными графами,
где узлы представляют атомы белка и лиганда, а связи включают как
внутримолекулярные, так и межмолекулярные взаимодействия.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import logging

from .egnn import EGNNLayer, EGNNConfig, EGNNModel

logger = logging.getLogger(__name__)


@dataclass
class ProteinLigandEGNNConfig:
    """Конфигурация для EGNN модели белок-лигандных комплексов."""
    
    # Размерности признаков
    protein_feature_dim: int = 8      # Размерность признаков атомов белка
    ligand_feature_dim: int = 8       # Размерность признаков атомов лиганда
    hidden_dim: int = 128             # Размерность скрытых слоев
    
    # Архитектура
    num_layers: int = 4               # Количество EGNN слоев
    num_attention_heads: int = 4      # Количество attention heads
    
    # Специфичные для белок-лиганда параметры
    interface_attention: bool = True   # Использовать attention для интерфейса
    separate_protein_ligand: bool = True  # Раздельная обработка белка и лиганда
    
    # Выходные параметры
    output_dim: int = 1               # Размерность выхода (аффинность)
    dropout: float = 0.1              # Dropout rate
    
    # Геометрические параметры
    update_coords: bool = False       # Обновлять координаты (обычно False для предсказания)
    coordinate_noise: float = 0.0     # Шум в координатах для регуляризации


class ProteinLigandAttention(nn.Module):
    """
    Attention механизм для белок-лигандных взаимодействий.
    
    Вычисляет attention веса между атомами белка и лиганда,
    учитывая как признаки атомов, так и их пространственное расположение.
    """
    
    def __init__(self, 
                 feature_dim: int,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        """
        Args:
            feature_dim: Размерность входных признаков
            num_heads: Количество attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        assert feature_dim % num_heads == 0, "feature_dim должен делиться на num_heads"
        
        # Проекции для query, key, value
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        
        # Проекция для расстояний
        self.distance_proj = nn.Linear(1, num_heads)
        
        # Выходная проекция
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(self, 
                protein_features: torch.Tensor,
                ligand_features: torch.Tensor,
                protein_coords: torch.Tensor,
                ligand_coords: torch.Tensor,
                interface_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Вычисляет attention между белком и лигандом.
        
        Args:
            protein_features: Признаки атомов белка [N_prot, feature_dim]
            ligand_features: Признаки атомов лиганда [N_lig, feature_dim]
            protein_coords: Координаты атомов белка [N_prot, 3]
            ligand_coords: Координаты атомов лиганда [N_lig, 3]
            interface_mask: Маска интерфейсных взаимодействий [N_prot, N_lig]
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Обновленные признаки белка и лиганда
        """
        N_prot, N_lig = protein_features.size(0), ligand_features.size(0)
        
        # Вычисляем расстояния между всеми парами атомов
        distances = torch.cdist(protein_coords, ligand_coords)  # [N_prot, N_lig]
        
        # Проекции для attention
        protein_q = self.query_proj(protein_features)  # [N_prot, feature_dim]
        ligand_k = self.key_proj(ligand_features)      # [N_lig, feature_dim]
        ligand_v = self.value_proj(ligand_features)    # [N_lig, feature_dim]
        
        # Reshape для multi-head attention
        protein_q = protein_q.view(N_prot, self.num_heads, self.head_dim)
        ligand_k = ligand_k.view(N_lig, self.num_heads, self.head_dim)
        ligand_v = ligand_v.view(N_lig, self.num_heads, self.head_dim)
        
        # Вычисляем attention scores
        attention_scores = torch.einsum('ihd,jhd->hij', protein_q, ligand_k) * self.scale
        
        # Добавляем информацию о расстояниях
        distance_bias = self.distance_proj(distances.unsqueeze(-1))  # [N_prot, N_lig, num_heads]
        # Правильная размерность для добавления: [N_prot, N_lig, num_heads]
        attention_scores = attention_scores + distance_bias
        
        # Применяем маску интерфейса, если предоставлена
        if interface_mask is not None:
            attention_scores = attention_scores.masked_fill(
                ~interface_mask.unsqueeze(-1), float('-inf')
            )
        
        # Softmax по лигандам для каждого атома белка
        attention_weights = F.softmax(attention_scores, dim=1)  # [N_prot, N_lig, num_heads]
        attention_weights = self.dropout(attention_weights)
        
        # Применяем attention к значениям лиганда
        attended_ligand = torch.einsum('ijh,jhd->ihd', attention_weights, ligand_v)
        attended_ligand = attended_ligand.reshape(N_prot, self.feature_dim)
        
        # Обновляем признаки белка
        updated_protein = self.output_proj(attended_ligand)
        
        # Аналогично для лиганда (attention к белку)
        ligand_q = self.query_proj(ligand_features)
        protein_k = self.key_proj(protein_features)
        protein_v = self.value_proj(protein_features)
        
        ligand_q = ligand_q.view(N_lig, self.num_heads, self.head_dim)
        protein_k = protein_k.view(N_prot, self.num_heads, self.head_dim)
        protein_v = protein_v.view(N_prot, self.num_heads, self.head_dim)
        
        attention_scores_lig = torch.einsum('ihd,jhd->hij', ligand_q, protein_k) * self.scale
        distance_bias_lig = self.distance_proj(distances.T.unsqueeze(-1))
        # Правильная размерность для добавления
        attention_scores_lig = attention_scores_lig + distance_bias_lig
        
        if interface_mask is not None:
            attention_scores_lig = attention_scores_lig.masked_fill(
                ~interface_mask.T.unsqueeze(-1), float('-inf')
            )
        
        attention_weights_lig = F.softmax(attention_scores_lig, dim=1)
        attention_weights_lig = self.dropout(attention_weights_lig)
        
        attended_protein = torch.einsum('ijh,jhd->ihd', attention_weights_lig, protein_v)
        attended_protein = attended_protein.reshape(N_lig, self.feature_dim)
        
        updated_ligand = self.output_proj(attended_protein)
        
        return updated_protein, updated_ligand


class ProteinLigandEGNNLayer(nn.Module):
    """
    EGNN слой для белок-лигандных комплексов.
    
    Расширяет базовый EGNN слой для работы с гетерогенными графами,
    включая специальную обработку интерфейсных взаимодействий.
    """
    
    def __init__(self, config: ProteinLigandEGNNConfig):
        super().__init__()
        
        self.config = config
        
        # Базовые EGNN слои для белка и лиганда
        egnn_config = EGNNConfig(
            node_feature_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_layers=1,
            output_dim=config.hidden_dim,
            update_coords=config.update_coords
        )
        
        self.protein_egnn = EGNNLayer(
            hidden_dim=config.hidden_dim,
            edge_feature_dim=0,
            update_coords=config.update_coords
        )
        self.ligand_egnn = EGNNLayer(
            hidden_dim=config.hidden_dim,
            edge_feature_dim=0,
            update_coords=config.update_coords
        )
        
        # Attention для интерфейсных взаимодействий
        if config.interface_attention:
            self.interface_attention = ProteinLigandAttention(
                feature_dim=config.hidden_dim,
                num_heads=config.num_attention_heads,
                dropout=config.dropout
            )
        
        # Нормализация
        self.protein_norm = nn.LayerNorm(config.hidden_dim)
        self.ligand_norm = nn.LayerNorm(config.hidden_dim)
    
    def forward(self, 
                protein_features: torch.Tensor,
                ligand_features: torch.Tensor,
                protein_coords: torch.Tensor,
                ligand_coords: torch.Tensor,
                protein_edges: torch.Tensor,
                ligand_edges: torch.Tensor,
                interface_edges: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass через слой.
        
        Args:
            protein_features: Признаки атомов белка [N_prot, hidden_dim]
            ligand_features: Признаки атомов лиганда [N_lig, hidden_dim]
            protein_coords: Координаты атомов белка [N_prot, 3]
            ligand_coords: Координаты атомов лиганда [N_lig, 3]
            protein_edges: Связи внутри белка [2, E_prot]
            ligand_edges: Связи внутри лиганда [2, E_lig]
            interface_edges: Интерфейсные связи [2, E_interface]
        
        Returns:
            Tuple: Обновленные признаки и координаты белка и лиганда
        """
        # Обработка внутримолекулярных взаимодействий
        protein_features_updated, protein_coords_updated = self.protein_egnn(
            protein_features, protein_coords, protein_edges
        )
        
        ligand_features_updated, ligand_coords_updated = self.ligand_egnn(
            ligand_features, ligand_coords, ligand_edges
        )
        
        # Обработка интерфейсных взаимодействий
        if self.config.interface_attention and interface_edges is not None:
            # Создаем маску интерфейса из связей
            interface_mask = self._create_interface_mask(
                protein_features.size(0), ligand_features.size(0), interface_edges
            )
            
            # Применяем attention
            protein_interface_update, ligand_interface_update = self.interface_attention(
                protein_features_updated, ligand_features_updated,
                protein_coords_updated, ligand_coords_updated,
                interface_mask
            )
            
            # Добавляем residual connections
            protein_features_updated = protein_features_updated + protein_interface_update
            ligand_features_updated = ligand_features_updated + ligand_interface_update
        
        # Нормализация
        protein_features_updated = self.protein_norm(protein_features_updated)
        ligand_features_updated = self.ligand_norm(ligand_features_updated)
        
        return (protein_features_updated, ligand_features_updated,
                protein_coords_updated, ligand_coords_updated)
    
    def _create_interface_mask(self, 
                              n_protein: int, 
                              n_ligand: int, 
                              interface_edges: torch.Tensor) -> torch.Tensor:
        """Создает маску интерфейсных взаимодействий из списка связей."""
        mask = torch.zeros(n_protein, n_ligand, dtype=torch.bool, device=interface_edges.device)
        
        # Предполагаем, что interface_edges содержит индексы [protein_idx, ligand_idx + n_protein]
        protein_indices = interface_edges[0]
        ligand_indices = interface_edges[1] - n_protein
        
        # Проверяем валидность индексов
        valid_mask = (protein_indices < n_protein) & (ligand_indices >= 0) & (ligand_indices < n_ligand)
        
        if valid_mask.any():
            mask[protein_indices[valid_mask], ligand_indices[valid_mask]] = True
        
        return mask


class ProteinLigandEGNN(nn.Module):
    """
    Полная EGNN модель для предсказания аффинности белок-лигандного связывания.
    
    Использует эквивариантную архитектуру для обработки 3D структур
    белок-лигандных комплексов и предсказания аффинности связывания.
    """
    
    def __init__(self, config: ProteinLigandEGNNConfig):
        super().__init__()
        
        self.config = config
        
        # Входные проекции
        self.protein_input_proj = nn.Linear(config.protein_feature_dim, config.hidden_dim)
        self.ligand_input_proj = nn.Linear(config.ligand_feature_dim, config.hidden_dim)
        
        # EGNN слои
        self.egnn_layers = nn.ModuleList([
            ProteinLigandEGNNLayer(config) for _ in range(config.num_layers)
        ])
        
        # Pooling для получения представлений молекул
        self.protein_pool = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        self.ligand_pool = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
        # Предсказание аффинности
        self.affinity_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, config.output_dim)
        )
        
        logger.info(f"Инициализирована ProteinLigandEGNN с {self._count_parameters():,} параметрами")
    
    def forward(self, 
                protein_features: torch.Tensor,
                ligand_features: torch.Tensor,
                protein_coords: torch.Tensor,
                ligand_coords: torch.Tensor,
                protein_edges: torch.Tensor,
                ligand_edges: torch.Tensor,
                interface_edges: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass модели.
        
        Args:
            protein_features: Признаки атомов белка [N_prot, protein_feature_dim]
            ligand_features: Признаки атомов лиганда [N_lig, ligand_feature_dim]
            protein_coords: Координаты атомов белка [N_prot, 3]
            ligand_coords: Координаты атомов лиганда [N_lig, 3]
            protein_edges: Связи внутри белка [2, E_prot]
            ligand_edges: Связи внутри лиганда [2, E_lig]
            interface_edges: Интерфейсные связи [2, E_interface]
        
        Returns:
            torch.Tensor: Предсказанная аффинность связывания [1]
        """
        # Входные проекции
        protein_h = self.protein_input_proj(protein_features)
        ligand_h = self.ligand_input_proj(ligand_features)
        
        protein_pos = protein_coords
        ligand_pos = ligand_coords
        
        # Добавляем шум к координатам для регуляризации (только во время обучения)
        if self.training and self.config.coordinate_noise > 0:
            protein_pos = protein_pos + torch.randn_like(protein_pos) * self.config.coordinate_noise
            ligand_pos = ligand_pos + torch.randn_like(ligand_pos) * self.config.coordinate_noise
        
        # Проход через EGNN слои
        for layer in self.egnn_layers:
            protein_h, ligand_h, protein_pos, ligand_pos = layer(
                protein_h, ligand_h, protein_pos, ligand_pos,
                protein_edges, ligand_edges, interface_edges
            )
        
        # Pooling для получения представлений молекул
        protein_repr = self.protein_pool(protein_h).mean(dim=0)  # Global average pooling
        ligand_repr = self.ligand_pool(ligand_h).mean(dim=0)
        
        # Объединяем представления
        complex_repr = torch.cat([protein_repr, ligand_repr], dim=-1)
        
        # Предсказываем аффинность
        affinity = self.affinity_predictor(complex_repr)
        
        return affinity
    
    def _count_parameters(self) -> int:
        """Подсчитывает количество параметров модели."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_num_parameters(self) -> int:
        """Возвращает количество параметров модели."""
        return self._count_parameters()


def create_protein_ligand_egnn(protein_feature_dim: int = 8,
                              ligand_feature_dim: int = 8,
                              hidden_dim: int = 128,
                              num_layers: int = 4,
                              **kwargs) -> ProteinLigandEGNN:
    """
    Создает EGNN модель для белок-лигандных комплексов.
    
    Args:
        protein_feature_dim: Размерность признаков атомов белка
        ligand_feature_dim: Размерность признаков атомов лиганда
        hidden_dim: Размерность скрытых слоев
        num_layers: Количество EGNN слоев
        **kwargs: Дополнительные параметры конфигурации
    
    Returns:
        ProteinLigandEGNN: Настроенная модель
    """
    config = ProteinLigandEGNNConfig(
        protein_feature_dim=protein_feature_dim,
        ligand_feature_dim=ligand_feature_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        **kwargs
    )
    
    return ProteinLigandEGNN(config)


# Пример использования
if __name__ == "__main__":
    # Создаем тестовые данные
    batch_size = 1
    n_protein_atoms = 100
    n_ligand_atoms = 20
    
    protein_features = torch.randn(n_protein_atoms, 8)
    ligand_features = torch.randn(n_ligand_atoms, 8)
    protein_coords = torch.randn(n_protein_atoms, 3)
    ligand_coords = torch.randn(n_ligand_atoms, 3)
    
    # Создаем случайные связи
    protein_edges = torch.randint(0, n_protein_atoms, (2, 150))
    ligand_edges = torch.randint(0, n_ligand_atoms, (2, 30))
    interface_edges = torch.stack([
        torch.randint(0, n_protein_atoms, (10,)),
        torch.randint(n_protein_atoms, n_protein_atoms + n_ligand_atoms, (10,))
    ])
    
    # Создаем модель
    model = create_protein_ligand_egnn()
    
    # Forward pass
    affinity = model(
        protein_features, ligand_features,
        protein_coords, ligand_coords,
        protein_edges, ligand_edges, interface_edges
    )
    
    print(f"Предсказанная аффинность: {affinity.item():.3f}")
    print(f"Количество параметров: {model.get_num_parameters():,}")