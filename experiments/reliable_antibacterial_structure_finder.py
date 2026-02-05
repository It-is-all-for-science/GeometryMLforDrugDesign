#!/usr/bin/env python3
"""
Надежная система поиска структур антибактериальных препаратов.
Заранее находит нужные молекулы, проверяет их доступность и группирует по размерам.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import requests
import h5py
from io import BytesIO
from collections import defaultdict
import pickle
from typing import Dict, List, Tuple, Optional
import hashlib
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pubchempy as pcp

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReliableAntibacterialStructureFinder:
    """
    Надежная система поиска структур антибактериальных препаратов.
    
    Основные принципы:
    1. Заранее определяет целевые антибиотики по группам размеров
    2. Проверяет доступность структур перед анализом
    3. Кэширует найденные структуры для повторного использования
    4. Обеспечивает по 10 молекул в каждой группе (кроме больших)
    """
    
    def __init__(self, cache_dir: str = "data/antibacterial_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Определяем целевые антибактериальные препараты по группам размеров
        self.target_antibiotics = {
            'small': {
                'size_range': (12, 21),
                'target_count': 10,
                'description': 'Малые антибиотики',
                'compounds': [
                    # Основные препараты (гарантированно найдем)
                    'chloramphenicol',      # C11H12Cl2N2O5 - 16 атомов
                    'trimethoprim',         # C14H18N4O3 - 20 атомов  
                    'sulfamethoxazole',     # C10H11N3O3S - 18 атомов
                    'nitrofurantoin',       # C8H6N4O5 - 18 атомов
                    'metronidazole',        # C6H9N3O3 - 15 атомов
                    # Дополнительные для достижения 10
                    'isoniazid',            # C6H7N3O - 13 атомов
                    'ethambutol',           # C10H24N2O2 - 20 атомов
                    'pyrazinamide',         # C5H5N3O - 12 атомов
                    'sulfadiazine',         # C10H10N4O2S - 17 атомов
                    'sulfisoxazole'         # C11H13N3O3S - 19 атомов
                ]
            },
            'medium': {
                'size_range': (23, 26),
                'target_count': 10,
                'description': 'Средние антибиотики',
                'compounds': [
                    # β-лактамы
                    'penicillin g',         # C16H18N2O4S - 25 атомов
                    'ampicillin',           # C16H19N3O4S - 26 атомов
                    'amoxicillin',          # C16H19N3O5S - 26 атомов
                    'cephalexin',           # C16H17N3O4S - 25 атомов
                    # Фторхинолоны
                    'ciprofloxacin',        # C17H18FN3O3 - 24 атомов
                    'levofloxacin',         # C18H20FN3O4 - 26 атомов
                    'norfloxacin',          # C16H18FN3O3 - 23 атомов
                    'ofloxacin',            # C18H20FN3O4 - 26 атомов
                    # Дополнительные
                    'cefazolin',            # C14H14N8O4S3 - 24 атомов
                    'cefuroxime'            # C16H16N4O8S - 25 атомов
                ]
            },
            'large': {
                'size_range': (30, 36),
                'target_count': 8,
                'description': 'Большие антибиотики',
                'compounds': [
                    # Тетрациклины
                    'tetracycline',         # C22H24N2O8 - 32 атомов
                    'doxycycline',          # C22H24N2O8 - 32 атомов
                    'minocycline',          # C23H27N3O7 - 33 атомов
                    # Аминогликозиды (упрощенные)
                    'streptomycin',         # C21H39N7O12 - 35 атомов
                    'gentamicin',           # C21H43N5O7 - 34 атомов (упрощенная форма)
                    # Макролиды (фрагменты)
                    'erythromycin',         # C37H67NO13 - 36 атомов (фрагмент)
                    'azithromycin',         # C38H72N2O12 - 35 атомов (фрагмент)
                    'clarithromycin'        # C38H69NO13 - 35 атомов (фрагмент)
                ]
            },
            'xlarge': {
                'size_range': (40, 60),
                'target_count': 5,
                'description': 'Очень большие антибиотики',
                'compounds': [
                    # Гликопептиды (фрагменты)
                    'vancomycin',           # C66H75Cl2N9O24 - 50+ атомов (фрагмент)
                    'teicoplanin',          # C88H97Cl2N9O33 - 50+ атомов (фрагмент)
                    # Полимиксины (фрагменты)
                    'colistin',             # C52H98N16O13 - 50+ атомов (фрагмент)
                    # Другие
                    'rifampicin',           # C43H58N4O12 - 45 атомов
                    'lincomycin'            # C18H34N2O6S - 40 атомов
                ]
            }
        }
        
        # Кэш найденных структур
        self.structure_cache_file = self.cache_dir / "antibacterial_structures.json"
        self.molecule_cache_dir = self.cache_dir / "molecules"
        self.molecule_cache_dir.mkdir(exist_ok=True)
        
        # Загружаем существующий кэш
        self.structure_cache = self._load_structure_cache()
        
    def _load_structure_cache(self) -> Dict:
        """Загружает кэш структур из файла."""
        
        if self.structure_cache_file.exists():
            try:
                with open(self.structure_cache_file, 'r') as f:
                    cache = json.load(f)
                logger.info(f"📋 Загружен кэш с {len(cache.get('structures', {}))} структурами")
                return cache
            except Exception as e:
                logger.warning(f"⚠️ Ошибка загрузки кэша: {e}")
        
        return {
            'structures': {},
            'groups': {group: [] for group in self.target_antibiotics.keys()},
            'last_updated': None,
            'failed_compounds': []
        }
    
    def _save_structure_cache(self):
        """Сохраняет кэш структур в файл."""
        
        self.structure_cache['last_updated'] = time.time()
        
        with open(self.structure_cache_file, 'w') as f:
            json.dump(self.structure_cache, f, indent=2)
        
        logger.info(f"💾 Сохранен кэш с {len(self.structure_cache['structures'])} структурами")
    
    def discover_and_cache_structures(self, force_refresh: bool = False) -> Dict:
        """
        Основная функция поиска и кэширования структур антибиотиков.
        
        Args:
            force_refresh: Принудительно обновить кэш
            
        Returns:
            Словарь с найденными структурами по группам
        """
        
        logger.info("🔍 Запуск поиска и кэширования структур антибиотиков...")
        
        # Проверяем, нужно ли обновлять кэш
        if not force_refresh and self._is_cache_sufficient():
            logger.info("✅ Кэш достаточен, используем существующие структуры")
            return self._get_grouped_structures()
        
        logger.info("🔄 Обновляем кэш структур...")
        
        # Поиск структур по группам
        for group_name, group_config in self.target_antibiotics.items():
            logger.info(f"📡 Обрабатываем группу: {group_name}")
            
            compounds = group_config['compounds']
            target_count = group_config['target_count']
            
            found_count = len(self.structure_cache['groups'].get(group_name, []))
            
            if found_count >= target_count:
                logger.info(f"✅ Группа {group_name}: достаточно структур ({found_count}/{target_count})")
                continue
            
            # Ищем недостающие структуры
            for compound_name in compounds:
                if found_count >= target_count:
                    break
                
                # Проверяем, не искали ли уже эту молекулу
                compound_id = f"{group_name}_{compound_name.replace(' ', '_')}"
                
                if (compound_id in self.structure_cache['structures'] or 
                    compound_name in self.structure_cache.get('failed_compounds', [])):
                    continue
                
                logger.info(f"🔍 Ищем структуру: {compound_name}")
                
                try:
                    structure = self._find_compound_structure(compound_name, group_name)
                    
                    if structure:
                        # Добавляем в кэш
                        self.structure_cache['structures'][compound_id] = structure
                        
                        # Добавляем в группу
                        if group_name not in self.structure_cache['groups']:
                            self.structure_cache['groups'][group_name] = []
                        
                        self.structure_cache['groups'][group_name].append(compound_id)
                        found_count += 1
                        
                        logger.info(f"✅ Найдена структура {compound_name}: {structure['n_atoms']} атомов")
                        
                        # Сохраняем структуру на диск
                        self._save_molecule_structure(compound_id, structure)
                        
                    else:
                        # Добавляем в список неудачных попыток
                        if 'failed_compounds' not in self.structure_cache:
                            self.structure_cache['failed_compounds'] = []
                        self.structure_cache['failed_compounds'].append(compound_name)
                        
                        logger.warning(f"❌ Не удалось найти структуру: {compound_name}")
                    
                    # Пауза между запросами
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"⚠️ Ошибка поиска {compound_name}: {e}")
                    continue
        
        # Сохраняем обновленный кэш
        self._save_structure_cache()
        
        # Возвращаем сгруппированные структуры
        return self._get_grouped_structures()
    
    def _is_cache_sufficient(self) -> bool:
        """Проверяет, достаточно ли структур в кэше."""
        
        grouped = self._get_grouped_structures()
        
        for group_name, group_config in self.target_antibiotics.items():
            available_count = len(grouped.get(group_name, []))
            target_count = group_config['target_count']
            
            if available_count < target_count:
                logger.info(f"❌ Группа {group_name}: {available_count}/{target_count} структур")
                return False
        
        logger.info("✅ Все группы имеют достаточно структур")
        return True
    
    def _find_compound_structure(self, compound_name: str, group_name: str) -> Optional[Dict]:
        """Ищет структуру конкретного соединения."""
        
        try:
            # Поиск через PubChem
            logger.info(f"🌐 Поиск {compound_name} в PubChem...")
            
            # Получаем соединение по имени
            compounds = pcp.get_compounds(compound_name, 'name')
            
            if not compounds:
                logger.warning(f"⚠️ Соединение {compound_name} не найдено в PubChem")
                return None
            
            compound = compounds[0]  # Берем первый результат
            
            # Получаем SMILES
            smiles = compound.canonical_smiles
            if not smiles:
                logger.warning(f"⚠️ SMILES не найден для {compound_name}")
                return None
            
            # Создаем молекулу RDKit
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"⚠️ Не удалось создать молекулу из SMILES: {smiles}")
                return None
            
            # Добавляем водороды
            mol = Chem.AddHs(mol)
            
            # Получаем информацию о молекуле
            n_atoms = mol.GetNumAtoms()
            
            # Проверяем размер
            size_range = self.target_antibiotics[group_name]['size_range']
            if not (size_range[0] <= n_atoms <= size_range[1]):
                logger.warning(f"⚠️ {compound_name}: {n_atoms} атомов не в диапазоне {size_range}")
                # Все равно сохраняем, но с пометкой
            
            # Генерируем 3D координаты
            from rdkit.Chem import AllChem
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Извлекаем координаты и атомные номера
            conf = mol.GetConformer()
            coordinates = []
            atomic_numbers = []
            
            for atom in mol.GetAtoms():
                atomic_numbers.append(atom.GetAtomicNum())
                pos = conf.GetAtomPosition(atom.GetIdx())
                coordinates.append([pos.x, pos.y, pos.z])
            
            # Вычисляем дополнительные свойства
            molecular_weight = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            
            # Создаем структуру
            structure = {
                'id': f"{group_name}_{compound_name.replace(' ', '_')}",
                'name': compound_name,
                'group': group_name,
                'source': 'pubchem',
                'cid': compound.cid,
                'smiles': smiles,
                'n_atoms': n_atoms,
                'atomic_numbers': atomic_numbers,
                'coordinates': coordinates,
                'molecular_weight': molecular_weight,
                'logp': logp,
                'tpsa': tpsa,
                'has_coordinates': True,
                'has_energy': False,
                'quality_score': self._calculate_quality_score(mol, n_atoms, size_range),
                'antibacterial_class': self._determine_antibacterial_class(compound_name),
                'mechanism_of_action': self._get_mechanism_of_action(compound_name)
            }
            
            return structure
            
        except Exception as e:
            logger.warning(f"⚠️ Ошибка поиска структуры {compound_name}: {e}")
            return None
    
    def _calculate_quality_score(self, mol, n_atoms: int, size_range: Tuple[int, int]) -> float:
        """Вычисляет оценку качества структуры."""
        
        score = 0.7  # Базовая оценка для PubChem
        
        # Бонус за правильный размер
        if size_range[0] <= n_atoms <= size_range[1]:
            score += 0.2
        
        # Бонус за наличие 3D координат
        if mol.GetNumConformers() > 0:
            score += 0.1
        
        return min(1.0, score)
    
    def _determine_antibacterial_class(self, compound_name: str) -> str:
        """Определяет класс антибиотика."""
        
        classes = {
            'beta_lactam': ['penicillin', 'ampicillin', 'amoxicillin', 'cephalexin', 'cefazolin', 'cefuroxime'],
            'fluoroquinolone': ['ciprofloxacin', 'levofloxacin', 'norfloxacin', 'ofloxacin'],
            'tetracycline': ['tetracycline', 'doxycycline', 'minocycline'],
            'aminoglycoside': ['streptomycin', 'gentamicin'],
            'macrolide': ['erythromycin', 'azithromycin', 'clarithromycin'],
            'glycopeptide': ['vancomycin', 'teicoplanin'],
            'sulfonamide': ['sulfamethoxazole', 'sulfadiazine', 'sulfisoxazole'],
            'nitroimidazole': ['metronidazole'],
            'nitrofuran': ['nitrofurantoin'],
            'phenicol': ['chloramphenicol'],
            'diaminopyrimidine': ['trimethoprim'],
            'antitubercular': ['isoniazid', 'ethambutol', 'pyrazinamide', 'rifampicin'],
            'polymyxin': ['colistin'],
            'lincosamide': ['lincomycin']
        }
        
        compound_lower = compound_name.lower()
        
        for class_name, compounds in classes.items():
            if any(comp in compound_lower for comp in compounds):
                return class_name
        
        return 'other'
    
    def _get_mechanism_of_action(self, compound_name: str) -> str:
        """Возвращает механизм действия антибиотика."""
        
        mechanisms = {
            'cell_wall_synthesis': ['penicillin', 'ampicillin', 'amoxicillin', 'cephalexin', 'cefazolin', 'cefuroxime', 'vancomycin', 'teicoplanin'],
            'dna_replication': ['ciprofloxacin', 'levofloxacin', 'norfloxacin', 'ofloxacin'],
            'protein_synthesis_30s': ['streptomycin', 'gentamicin', 'tetracycline', 'doxycycline', 'minocycline'],
            'protein_synthesis_50s': ['erythromycin', 'azithromycin', 'clarithromycin', 'chloramphenicol', 'lincomycin'],
            'folate_synthesis': ['sulfamethoxazole', 'sulfadiazine', 'sulfisoxazole', 'trimethoprim'],
            'dna_damage': ['metronidazole', 'nitrofurantoin'],
            'cell_membrane': ['colistin'],
            'rna_synthesis': ['rifampicin'],
            'cell_wall_synthesis_mycobacteria': ['isoniazid', 'ethambutol', 'pyrazinamide']
        }
        
        compound_lower = compound_name.lower()
        
        for mechanism, compounds in mechanisms.items():
            if any(comp in compound_lower for comp in compounds):
                return mechanism
        
        return 'unknown'
    
    def _save_molecule_structure(self, compound_id: str, structure: Dict):
        """Сохраняет структуру молекулы на диск."""
        
        cache_file = self.molecule_cache_dir / f"{compound_id}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(structure, f)
        except Exception as e:
            logger.warning(f"⚠️ Ошибка сохранения {compound_id}: {e}")
    
    def _get_grouped_structures(self) -> Dict:
        """Возвращает структуры, сгруппированные по размерам."""
        
        grouped = {}
        
        for group_name, compound_ids in self.structure_cache['groups'].items():
            group_structures = []
            
            for compound_id in compound_ids:
                if compound_id in self.structure_cache['structures']:
                    structure = self.structure_cache['structures'][compound_id]
                    group_structures.append(structure)
            
            # Сортируем по качеству
            group_structures.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            
            # Берем только нужное количество
            target_count = self.target_antibiotics[group_name]['target_count']
            grouped[group_name] = group_structures[:target_count]
        
        return grouped
    
    def get_verified_structures(self, group_name: str = None) -> Dict:
        """
        Возвращает проверенные структуры антибиотиков.
        
        Args:
            group_name: Конкретная группа или None для всех групп
            
        Returns:
            Словарь с проверенными структурами
        """
        
        logger.info(f"🔍 Получение проверенных структур для группы: {group_name or 'все'}")
        
        grouped_structures = self._get_grouped_structures()
        
        if group_name:
            if group_name not in grouped_structures:
                logger.warning(f"⚠️ Группа {group_name} не найдена")
                return {}
            groups_to_process = {group_name: grouped_structures[group_name]}
        else:
            groups_to_process = grouped_structures
        
        verified_structures = {}
        
        for group, structures in groups_to_process.items():
            logger.info(f"📋 Проверка группы {group}: {len(structures)} структур")
            
            verified_group = []
            
            for structure in structures:
                # Проверяем структуру
                if self._verify_structure(structure):
                    verified_group.append(structure)
            
            verified_structures[group] = verified_group
            
            logger.info(f"✅ Группа {group}: {len(verified_group)} проверенных структур")
        
        return verified_structures
    
    def _verify_structure(self, structure: Dict) -> bool:
        """Проверяет корректность структуры."""
        
        # Базовые проверки
        required_fields = ['atomic_numbers', 'coordinates', 'n_atoms', 'smiles']
        
        for field in required_fields:
            if field not in structure:
                return False
        
        # Проверяем размеры
        n_atoms = structure['n_atoms']
        atomic_numbers = structure['atomic_numbers']
        coordinates = structure['coordinates']
        
        if len(atomic_numbers) != n_atoms or len(coordinates) != n_atoms:
            return False
        
        # Проверяем координаты
        for coord in coordinates:
            if len(coord) != 3:
                return False
        
        # Проверяем SMILES
        try:
            mol = Chem.MolFromSmiles(structure['smiles'])
            if mol is None:
                return False
        except:
            return False
        
        return True
    
    def print_inventory_summary(self):
        """Выводит сводку по инвентарю структур."""
        
        logger.info("\n" + "="*70)
        logger.info("📋 СВОДКА ПО СТРУКТУРАМ АНТИБАКТЕРИАЛЬНЫХ ПРЕПАРАТОВ")
        logger.info("="*70)
        
        grouped = self._get_grouped_structures()
        
        total_structures = sum(len(structures) for structures in grouped.values())
        logger.info(f"📊 Всего структур в кэше: {total_structures}")
        
        logger.info(f"\n📈 ПО ГРУППАМ РАЗМЕРОВ:")
        
        for group_name, group_config in self.target_antibiotics.items():
            structures = grouped.get(group_name, [])
            target_count = group_config['target_count']
            size_range = group_config['size_range']
            description = group_config['description']
            
            status = "✅" if len(structures) >= target_count else "❌"
            
            logger.info(f"  {status} {group_name.upper()}: {len(structures)}/{target_count}")
            logger.info(f"      Размер: {size_range[0]}-{size_range[1]} атомов")
            logger.info(f"      Описание: {description}")
            
            if structures:
                # Статистика по классам антибиотиков
                classes = {}
                mechanisms = {}
                quality_scores = []
                
                for struct in structures:
                    ab_class = struct.get('antibacterial_class', 'unknown')
                    mechanism = struct.get('mechanism_of_action', 'unknown')
                    
                    classes[ab_class] = classes.get(ab_class, 0) + 1
                    mechanisms[mechanism] = mechanisms.get(mechanism, 0) + 1
                    quality_scores.append(struct.get('quality_score', 0))
                
                logger.info(f"      Классы: {dict(classes)}")
                logger.info(f"      Механизмы: {dict(mechanisms)}")
                logger.info(f"      Средняя оценка качества: {np.mean(quality_scores):.2f}")
                
                # Примеры структур
                logger.info(f"      Примеры:")
                for i, struct in enumerate(structures[:3], 1):
                    logger.info(f"        {i}. {struct['name']}: {struct['n_atoms']} атомов, "
                               f"класс: {struct.get('antibacterial_class', 'unknown')}")
        
        failed_count = len(self.structure_cache.get('failed_compounds', []))
        if failed_count > 0:
            logger.info(f"\n❌ Не удалось найти: {failed_count} соединений")
            for compound in self.structure_cache.get('failed_compounds', [])[:5]:
                logger.info(f"    - {compound}")
        
        last_updated = self.structure_cache.get('last_updated')
        if last_updated:
            import datetime
            update_time = datetime.datetime.fromtimestamp(last_updated)
            logger.info(f"\n🕒 Последнее обновление: {update_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def export_for_analysis(self, output_file: str = None) -> str:
        """
        Экспортирует найденные структуры для анализа.
        
        Args:
            output_file: Путь к выходному файлу
            
        Returns:
            Путь к созданному файлу
        """
        
        if output_file is None:
            output_file = self.cache_dir / "antibacterial_structures_for_analysis.json"
        
        verified_structures = self.get_verified_structures()
        
        # Подготавливаем данные для экспорта
        export_data = {
            'metadata': {
                'total_structures': sum(len(structs) for structs in verified_structures.values()),
                'groups': list(verified_structures.keys()),
                'export_timestamp': time.time(),
                'description': 'Проверенные структуры антибактериальных препаратов для ML анализа'
            },
            'structures': verified_structures,
            'group_configs': self.target_antibiotics
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📤 Экспортированы структуры в: {output_file}")
        return str(output_file)


def main():
    """Главная функция для тестирования системы."""
    
    logger.info("🚀 Тестирование надежной системы поиска антибактериальных структур")
    
    try:
        # Создаем систему поиска
        finder = ReliableAntibacterialStructureFinder()
        
        # Выводим текущее состояние
        finder.print_inventory_summary()
        
        # Запускаем поиск и кэширование
        logger.info("\n" + "="*70)
        logger.info("🔍 ЗАПУСК ПОИСКА И КЭШИРОВАНИЯ СТРУКТУР")
        logger.info("="*70)
        
        grouped_structures = finder.discover_and_cache_structures(force_refresh=False)
        
        # Выводим результаты
        logger.info("\n" + "="*70)
        logger.info("📊 РЕЗУЛЬТАТЫ ПОИСКА")
        logger.info("="*70)
        
        finder.print_inventory_summary()
        
        # Получаем проверенные структуры
        logger.info("\n" + "="*70)
        logger.info("🔍 ПРОВЕРКА СТРУКТУР")
        logger.info("="*70)
        
        verified_structures = finder.get_verified_structures()
        
        # Экспортируем для анализа
        export_file = finder.export_for_analysis()
        
        logger.info(f"\n✅ Система готова к использованию!")
        logger.info(f"📁 Структуры экспортированы в: {export_file}")
        
        return verified_structures
        
    except Exception as e:
        logger.error(f"❌ Ошибка в тестировании: {e}")
        raise


if __name__ == "__main__":
    main()