"""
Модуль для парсинга PDB файлов и извлечения белок-лигандных комплексов.

Содержит функции для чтения структур белков и лигандов из PDB файлов,
извлечения координат атомов, типов атомов и других структурных данных.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, NamedTuple
from pathlib import Path
import logging
from dataclasses import dataclass
import re

logger = logging.getLogger(__name__)


@dataclass
class AtomInfo:
    """Информация об атоме из PDB файла."""
    atom_id: int
    atom_name: str
    residue_name: str
    chain_id: str
    residue_id: int
    coordinates: np.ndarray  # [3] - x, y, z
    element: str
    occupancy: float
    b_factor: float


@dataclass
class ProteinStructure:
    """Структура белка."""
    atoms: List[AtomInfo]
    chains: Dict[str, List[AtomInfo]]
    residues: Dict[Tuple[str, int], List[AtomInfo]]  # (chain_id, residue_id) -> atoms
    
    def get_coordinates(self) -> np.ndarray:
        """Возвращает координаты всех атомов."""
        return np.array([atom.coordinates for atom in self.atoms])
    
    def get_elements(self) -> List[str]:
        """Возвращает элементы всех атомов."""
        return [atom.element for atom in self.atoms]
    
    def get_ca_coordinates(self) -> np.ndarray:
        """Возвращает координаты только CA атомов."""
        ca_coords = []
        for atom in self.atoms:
            if atom.atom_name.strip() == 'CA':
                ca_coords.append(atom.coordinates)
        return np.array(ca_coords)


@dataclass
class LigandStructure:
    """Структура лиганда."""
    atoms: List[AtomInfo]
    molecule_name: str
    
    def get_coordinates(self) -> np.ndarray:
        """Возвращает координаты всех атомов."""
        return np.array([atom.coordinates for atom in self.atoms])
    
    def get_elements(self) -> List[str]:
        """Возвращает элементы всех атомов."""
        return [atom.element for atom in self.atoms]


class PDBParser:
    """Парсер для PDB файлов."""
    
    # Стандартные аминокислоты
    STANDARD_RESIDUES = {
        'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
    }
    
    # Маппинг элементов по атомным именам
    ELEMENT_MAPPING = {
        'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'CD': 'C', 'CE': 'C', 'CZ': 'C',
        'N': 'N', 'ND': 'N', 'NE': 'N', 'NH': 'N', 'NZ': 'N',
        'O': 'O', 'OD': 'O', 'OE': 'O', 'OG': 'O', 'OH': 'O',
        'S': 'S', 'SD': 'S', 'SG': 'S',
        'P': 'P',
        'H': 'H'
    }
    
    def __init__(self):
        """Инициализация парсера."""
        pass
    
    def parse_pdb_file(self, pdb_path: str) -> Tuple[ProteinStructure, List[LigandStructure]]:
        """
        Парсит PDB файл и извлекает структуры белка и лигандов.
        
        Args:
            pdb_path: Путь к PDB файлу
        
        Returns:
            Tuple[ProteinStructure, List[LigandStructure]]: Белок и лиганды
        """
        pdb_path = Path(pdb_path)
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB файл не найден: {pdb_path}")
        
        protein_atoms = []
        ligand_atoms_by_residue = {}
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    atom_info = self._parse_atom_line(line)
                    
                    if atom_info is None:
                        continue
                    
                    # Определяем, является ли это частью белка или лиганда
                    if self._is_protein_atom(atom_info):
                        protein_atoms.append(atom_info)
                    else:
                        # Группируем атомы лиганда по остаткам
                        residue_key = (atom_info.chain_id, atom_info.residue_id, atom_info.residue_name)
                        if residue_key not in ligand_atoms_by_residue:
                            ligand_atoms_by_residue[residue_key] = []
                        ligand_atoms_by_residue[residue_key].append(atom_info)
        
        # Создаем структуру белка
        protein = self._create_protein_structure(protein_atoms)
        
        # Создаем структуры лигандов
        ligands = []
        for (chain_id, residue_id, residue_name), atoms in ligand_atoms_by_residue.items():
            # Фильтруем воду и ионы
            if not self._is_relevant_ligand(residue_name, atoms):
                continue
                
            ligand = LigandStructure(
                atoms=atoms,
                molecule_name=residue_name
            )
            ligands.append(ligand)
        
        logger.info(f"Парсинг {pdb_path.name}: {len(protein_atoms)} атомов белка, "
                   f"{len(ligands)} лигандов")
        
        return protein, ligands
    
    def _parse_atom_line(self, line: str) -> Optional[AtomInfo]:
        """Парсит строку ATOM/HETATM из PDB файла."""
        try:
            # PDB формат фиксированной ширины
            atom_id = int(line[6:11].strip())
            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip()
            chain_id = line[21:22].strip()
            residue_id = int(line[22:26].strip())
            
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            coordinates = np.array([x, y, z])
            
            occupancy = float(line[54:60].strip()) if line[54:60].strip() else 1.0
            b_factor = float(line[60:66].strip()) if line[60:66].strip() else 0.0
            
            # Определяем элемент
            element = line[76:78].strip() if len(line) > 76 else ''
            if not element:
                element = self._guess_element_from_atom_name(atom_name)
            
            return AtomInfo(
                atom_id=atom_id,
                atom_name=atom_name,
                residue_name=residue_name,
                chain_id=chain_id,
                residue_id=residue_id,
                coordinates=coordinates,
                element=element,
                occupancy=occupancy,
                b_factor=b_factor
            )
            
        except (ValueError, IndexError) as e:
            logger.warning(f"Ошибка парсинга строки PDB: {line.strip()}, ошибка: {e}")
            return None
    
    def _is_protein_atom(self, atom_info: AtomInfo) -> bool:
        """Определяет, является ли атом частью белка."""
        return atom_info.residue_name in self.STANDARD_RESIDUES
    
    def _is_relevant_ligand(self, residue_name: str, atoms: List[AtomInfo]) -> bool:
        """Определяет, является ли лиганд релевантным (не вода, не ион)."""
        # Исключаем воду
        if residue_name in ['HOH', 'WAT', 'H2O']:
            return False
        
        # Исключаем простые ионы
        if residue_name in ['NA', 'CL', 'K', 'MG', 'CA', 'ZN', 'FE', 'MN']:
            return False
        
        # Исключаем очень маленькие молекулы (< 5 атомов)
        if len(atoms) < 5:
            return False
        
        return True
    
    def _guess_element_from_atom_name(self, atom_name: str) -> str:
        """Угадывает элемент по имени атома."""
        # Убираем цифры и специальные символы
        clean_name = re.sub(r'[0-9\'\"]', '', atom_name).upper()
        
        # Проверяем точные совпадения
        for pattern, element in self.ELEMENT_MAPPING.items():
            if clean_name.startswith(pattern):
                return element
        
        # Если не найдено, берем первую букву
        if clean_name:
            return clean_name[0]
        
        return 'C'  # По умолчанию углерод
    
    def _create_protein_structure(self, atoms: List[AtomInfo]) -> ProteinStructure:
        """Создает структуру белка из списка атомов."""
        chains = {}
        residues = {}
        
        for atom in atoms:
            # Группировка по цепям
            if atom.chain_id not in chains:
                chains[atom.chain_id] = []
            chains[atom.chain_id].append(atom)
            
            # Группировка по остаткам
            residue_key = (atom.chain_id, atom.residue_id)
            if residue_key not in residues:
                residues[residue_key] = []
            residues[residue_key].append(atom)
        
        return ProteinStructure(
            atoms=atoms,
            chains=chains,
            residues=residues
        )


class ProteinFeatureExtractor:
    """Извлекает признаки из структуры белка."""
    
    # Физико-химические свойства аминокислот
    RESIDUE_PROPERTIES = {
        'ALA': [0, 0, 0, 0, 0],  # [hydrophobic, polar, positive, negative, aromatic]
        'ARG': [0, 1, 1, 0, 0],
        'ASN': [0, 1, 0, 0, 0],
        'ASP': [0, 1, 0, 1, 0],
        'CYS': [0, 1, 0, 0, 0],
        'GLN': [0, 1, 0, 0, 0],
        'GLU': [0, 1, 0, 1, 0],
        'GLY': [0, 0, 0, 0, 0],
        'HIS': [0, 1, 1, 0, 1],
        'ILE': [1, 0, 0, 0, 0],
        'LEU': [1, 0, 0, 0, 0],
        'LYS': [0, 1, 1, 0, 0],
        'MET': [1, 0, 0, 0, 0],
        'PHE': [1, 0, 0, 0, 1],
        'PRO': [0, 0, 0, 0, 0],
        'SER': [0, 1, 0, 0, 0],
        'THR': [0, 1, 0, 0, 0],
        'TRP': [1, 0, 0, 0, 1],
        'TYR': [0, 1, 0, 0, 1],
        'VAL': [1, 0, 0, 0, 0]
    }
    
    def extract_residue_features(self, protein: ProteinStructure) -> torch.Tensor:
        """
        Извлекает признаки остатков белка.
        
        Args:
            protein: Структура белка
        
        Returns:
            torch.Tensor: Признаки остатков [N_residues, feature_dim]
        """
        features = []
        
        # Группируем атомы по остаткам
        residue_keys = sorted(protein.residues.keys())
        
        for chain_id, residue_id in residue_keys:
            residue_atoms = protein.residues[(chain_id, residue_id)]
            
            if not residue_atoms:
                continue
            
            # Базовые признаки остатка
            residue_name = residue_atoms[0].residue_name
            base_features = self.RESIDUE_PROPERTIES.get(residue_name, [0, 0, 0, 0, 0])
            
            # Геометрические признаки
            coords = np.array([atom.coordinates for atom in residue_atoms])
            center = coords.mean(axis=0)
            
            # Дополнительные признаки
            additional_features = [
                len(residue_atoms),  # количество атомов
                np.linalg.norm(center),  # расстояние от начала координат
                coords.std(),  # разброс координат атомов
            ]
            
            # Объединяем все признаки
            residue_features = base_features + additional_features
            features.append(residue_features)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def extract_atom_features(self, atoms: List[AtomInfo]) -> torch.Tensor:
        """
        Извлекает признаки атомов.
        
        Args:
            atoms: Список атомов
        
        Returns:
            torch.Tensor: Признаки атомов [N_atoms, feature_dim]
        """
        # Маппинг элементов к индексам
        element_to_idx = {
            'C': 0, 'N': 1, 'O': 2, 'S': 3, 'P': 4, 'H': 5
        }
        
        features = []
        
        for atom in atoms:
            # One-hot кодирование элемента
            element_features = [0] * len(element_to_idx)
            if atom.element in element_to_idx:
                element_features[element_to_idx[atom.element]] = 1
            
            # Дополнительные признаки
            additional_features = [
                atom.b_factor / 100.0,  # нормализованный B-фактор
                atom.occupancy,  # заполненность
            ]
            
            atom_features = element_features + additional_features
            features.append(atom_features)
        
        return torch.tensor(features, dtype=torch.float32)


def create_mock_pdbbind_data(num_complexes: int = 100) -> Tuple[List, torch.Tensor, Dict]:
    """
    Создает синтетические данные в формате PDBbind для тестирования.
    
    Args:
        num_complexes: Количество комплексов для генерации
    
    Returns:
        Tuple: Список комплексов, аффинности, метаданные
    """
    from .loaders import ProteinComplexData
    
    complexes = []
    affinities = []
    
    np.random.seed(42)  # For reproducibility
    
    for i in range(num_complexes):
        # Генерируем случайные размеры
        n_protein_atoms = np.random.randint(50, 200)
        n_ligand_atoms = np.random.randint(10, 50)
        
        # Создаем случайные координаты
        protein_coords = torch.randn(n_protein_atoms, 3) * 10
        ligand_coords = torch.randn(n_ligand_atoms, 3) * 5
        
        # Размещаем лиганд рядом с белком
        ligand_coords += torch.randn(3) * 2
        
        # Создаем признаки
        protein_features = torch.randn(n_protein_atoms, 8)  # 8 признаков на атом
        ligand_features = torch.randn(n_ligand_atoms, 8)
        
        # Создаем интерфейсные связи
        n_interface_edges = np.random.randint(5, 20)
        protein_indices = torch.randint(0, n_protein_atoms, (n_interface_edges,))
        ligand_indices = torch.randint(0, n_ligand_atoms, (n_interface_edges,)) + n_protein_atoms
        interface_edges = torch.stack([protein_indices, ligand_indices])
        
        # Генерируем аффинность (pKd от 4 до 12)
        affinity = torch.tensor(np.random.uniform(4.0, 12.0))
        
        complex_data = ProteinComplexData(
            protein_coords=protein_coords,
            ligand_coords=ligand_coords,
            protein_features=protein_features,
            ligand_features=ligand_features,
            interface_edges=interface_edges,
            binding_affinity=affinity
        )
        
        complexes.append(complex_data)
        affinities.append(affinity)
    
    affinities_tensor = torch.stack(affinities)
    
    metadata = {
        'dataset_name': 'PDBbind_Mock',
        'num_complexes': num_complexes,
        'affinity_mean': affinities_tensor.mean().item(),
        'affinity_std': affinities_tensor.std().item(),
        'affinity_range': (affinities_tensor.min().item(), affinities_tensor.max().item()),
        'note': 'Синтетические данные для тестирования'
    }
    
    logger.info(f"Создано {num_complexes} синтетических PDBbind комплексов")
    
    return complexes, affinities_tensor, metadata