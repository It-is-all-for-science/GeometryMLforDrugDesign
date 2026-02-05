#!/usr/bin/env python3
"""
Task 31: Валидация на реальных экспериментальных данных для антибактериальных препаратов

Этот скрипт выполняет полную валидацию лучшей EGNN Model 3 на экспериментальных данных
антибактериальных препаратов с анализом domain shift и uncertainty quantification.

Subtasks:
31.1 ✅ Поиск экспериментальных HOMO-LUMO Gap данных (завершено)
31.2 🔄 Предсказания Gap энергий лучшей EGNN Model 3
31.3 🔄 Статистическое сравнение с экспериментальными данными
31.4 🔄 Comprehensive визуализации и отчет
31.5 🔄 Интеграция с существующими результатами антибактериального анализа
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к нашим модулям
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit не установлен. Некоторые функции будут недоступны.")

from step_03_models.egnn import EGNNModel, EGNNConfig
from step_01_data.loaders import MolecularDataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Task31ExperimentalValidator:
    """
    Валидатор для Task 31 - экспериментальная валидация EGNN моделей
    на антибактериальных препаратах.
    """
    
    def __init__(self):
        self.results_dir = Path("results/experimental_gap_validation")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Пути к моделям (исправляем пути)
        self.model_paths = {
            "egnn_model3": Path("results/improved_egnn_ensemble/models/improved_egnn_model3/best_model.pth"),
            "egnn_model1": Path("results/improved_egnn_ensemble/models/improved_egnn_model1/best_model.pth"),
            "egnn_model2": Path("results/improved_egnn_ensemble/models/improved_egnn_model2/best_model.pth")
        }
        
        # Загружаем экспериментальные данные
        self.experimental_data = self._load_experimental_data()
        
        # Результаты предсказаний
        self.predictions = {}
        self.ensemble_predictions = {}
        self.uncertainty_estimates = {}
        
        # Статистические результаты
        self.validation_metrics = {}
        self.domain_shift_analysis = {}
        
    def _load_experimental_data(self) -> Dict:
        """Загружает финальные экспериментальные данные."""
        
        # Пробуем загрузить финальный датасет
        data_file = Path("results/experimental_gap_validation/final_experimental_dataset.json")
        
        if not data_file.exists():
            # Если финального нет, пробуем полный
            data_file = Path("results/experimental_gap_validation/complete_experimental_dataset.json")
            
        if not data_file.exists():
            # Если и полного нет, пробуем расширенный
            data_file = Path("results/experimental_gap_validation/expanded_experimental_dataset.json")
            
        if not data_file.exists():
            # Если и расширенного нет, пробуем обновленный
            data_file = Path("results/experimental_gap_validation/updated_experimental_gap_dataset.json")
        
        if not data_file.exists():
            raise FileNotFoundError(f"Экспериментальные данные не найдены")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"📋 Загружены экспериментальные данные: {data['metadata']['total_molecules']} молекул из {data_file.name}")
        return data
    
    def _load_egnn_model(self, model_path: Path) -> EGNNModel:
        """Загружает EGNN модель из checkpoint."""
        
        if not model_path.exists():
            raise FileNotFoundError(f"Модель не найдена: {model_path}")
        
        # Загружаем checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Создаем модель с правильной конфигурацией (5 слоёв, hidden_dim=256, как в checkpoint)
        config = EGNNConfig(
            node_feature_dim=11,  # Стандартные атомные признаки
            edge_feature_dim=0,   # 0 как в checkpoint (не 4!)
            hidden_dim=256,       # 256 как в checkpoint (не 128!)
            num_layers=5,         # 5 слоёв как в checkpoint
            output_dim=1,         # HOMO-LUMO Gap
            dropout=0.1,
            attention=True,
            normalize=True,
            update_coords=False   # Не обновляем координаты
        )
        
        model = EGNNModel(config)
        
        # Загружаем веса с правильным префиксом
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Убираем префикс "egnn_model." из ключей
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('egnn_model.'):
                new_key = key[11:]  # Убираем "egnn_model."
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        
        model.load_state_dict(new_state_dict)
        model.eval()
        logger.info(f"✅ Модель загружена: {model_path.name}")
        
        return model
    
    def _smiles_to_graph(self, smiles: str) -> Optional[Dict]:
        """Конвертирует SMILES в молекулярный граф для EGNN."""
        
        if not RDKIT_AVAILABLE:
            logger.error("RDKit не установлен. Невозможно конвертировать SMILES.")
            return None
        
        try:
            # Создаем молекулу из SMILES
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Не удалось создать молекулу из SMILES: {smiles}")
                return None
            
            # Добавляем водороды
            mol = Chem.AddHs(mol)
            
            # Генерируем 3D координаты
            AllChem.EmbedMolecule(mol, randomSeed=42)
            AllChem.MMFFOptimizeMolecule(mol)
            
            # Извлекаем атомные признаки
            node_features = []
            coordinates = []
            
            for atom in mol.GetAtoms():
                # Базовые атомные признаки (11 признаков)
                features = [
                    atom.GetAtomicNum(),                    # Атомный номер
                    atom.GetDegree(),                       # Степень
                    atom.GetFormalCharge(),                 # Формальный заряд
                    atom.GetHybridization().real,           # Гибридизация
                    int(atom.GetIsAromatic()),              # Ароматичность
                    atom.GetNumRadicalElectrons(),          # Радикальные электроны
                    int(atom.IsInRing()),                   # В кольце
                    atom.GetMass(),                         # Атомная масса
                    atom.GetTotalValence(),                 # Валентность
                    atom.GetNumImplicitHs(),                # Неявные водороды
                    int(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED)  # Хиральность
                ]
                node_features.append(features)
                
                # Координаты
                pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                coordinates.append([pos.x, pos.y, pos.z])
            
            # Извлекаем связи (без признаков связей, так как модель обучена без них)
            edge_indices = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                # Добавляем связь в обе стороны (неориентированный граф)
                edge_indices.extend([[i, j], [j, i]])
            
            # Если нет связей, создаем пустые тензоры
            if not edge_indices:
                edge_indices = [[], []]
            
            # Конвертируем в numpy массивы
            node_features = np.array(node_features, dtype=np.float32)
            coordinates = np.array(coordinates, dtype=np.float32)
            edge_index = np.array(edge_indices, dtype=np.int64).T  # [2, E]
            edge_features = np.zeros((len(edge_indices), 0), dtype=np.float32)  # Пустые признаки связей
            
            return {
                'node_features': node_features,
                'coordinates': coordinates,
                'edge_index': edge_index,
                'edge_features': edge_features,
                'n_atoms': len(node_features)
            }
            
        except Exception as e:
            logger.warning(f"Ошибка конвертации SMILES {smiles}: {e}")
            return None
    
    def _prepare_molecular_data(self, molecules: List[Dict]) -> List[Dict]:
        """Подготавливает молекулярные данные для предсказаний."""
        
        logger.info("🔄 Подготовка молекулярных данных...")
        
        prepared_molecules = []
        
        for mol_data in molecules:
            if not mol_data.get('smiles'):
                logger.warning(f"Пропускаем {mol_data['name']}: нет SMILES")
                continue
            
            try:
                # Конвертируем SMILES в молекулярный граф
                mol_graph = self._smiles_to_graph(mol_data['smiles'])
                
                if mol_graph is None:
                    logger.warning(f"Не удалось создать граф для {mol_data['name']}")
                    continue
                
                # Добавляем экспериментальные данные
                mol_graph.update({
                    'name': mol_data['name'],
                    'experimental_gap': mol_data.get('gap_energy'),
                    'n_atoms': mol_data.get('n_atoms'),
                    'quality_score': mol_data.get('quality_score', 0.5),
                    'antibacterial_class': mol_data.get('antibacterial_class'),
                    'source': mol_data.get('source'),
                    'method': mol_data.get('method')
                })
                
                prepared_molecules.append(mol_graph)
                
            except Exception as e:
                logger.warning(f"Ошибка обработки {mol_data['name']}: {e}")
                continue
        
        logger.info(f"✅ Подготовлено {len(prepared_molecules)} молекул для предсказаний")
        return prepared_molecules
    
    def _predict_with_model(self, model: EGNNModel, molecules: List[Dict]) -> Dict[str, float]:
        """Делает предсказания с одной моделью."""
        
        predictions = {}
        
        with torch.no_grad():
            for mol_data in molecules:
                try:
                    # Подготавливаем входные данные
                    node_features = torch.tensor(mol_data['node_features'], dtype=torch.float32)
                    coordinates = torch.tensor(mol_data['coordinates'], dtype=torch.float32)
                    edge_index = torch.tensor(mol_data['edge_index'], dtype=torch.long)
                    edge_features = torch.tensor(mol_data['edge_features'], dtype=torch.float32)
                    
                    # Проверяем размерности
                    if edge_index.numel() == 0:
                        # Если нет связей, создаем пустой edge_index
                        edge_index = torch.zeros((2, 0), dtype=torch.long)
                    
                    # Не используем edge_features, так как модель обучена без них
                    edge_features = None
                    
                    # Делаем предсказание
                    results = model(
                        x=node_features,
                        pos=coordinates,
                        edge_index=edge_index,
                        edge_attr=edge_features,
                        batch=None  # Одна молекула
                    )
                    
                    prediction = results['prediction'].item()
                    predictions[mol_data['name']] = prediction
                    
                except Exception as e:
                    logger.warning(f"Ошибка предсказания для {mol_data['name']}: {e}")
                    continue
        
        return predictions
    
    def run_subtask_31_2(self):
        """
        Subtask 31.2: Предсказания Gap энергий лучшей EGNN Model 3
        """
        
        logger.info("🚀 SUBTASK 31.2: ПРЕДСКАЗАНИЯ GAP ЭНЕРГИЙ ЛУЧШЕЙ EGNN MODEL 3")
        logger.info("="*80)
        
        try:
            # 1. Подготовка молекулярных данных
            logger.info("\n📋 Подготовка экспериментальных молекул...")
            
            molecules = self.experimental_data['molecules']
            # Фильтруем только молекулы с экспериментальными Gap значениями
            valid_molecules = [mol for mol in molecules if mol.get('gap_energy') is not None]
            
            logger.info(f"📊 Молекул с экспериментальными Gap: {len(valid_molecules)}")
            
            prepared_molecules = self._prepare_molecular_data(valid_molecules)
            
            if not prepared_molecules:
                raise ValueError("Не удалось подготовить молекулы для предсказаний")
            
            # 2. Загрузка лучшей модели (Model 3)
            logger.info("\n🤖 Загрузка лучшей EGNN Model 3...")
            
            best_model = self._load_egnn_model(self.model_paths["egnn_model3"])
            
            # 3. Предсказания с лучшей моделью
            logger.info("\n🔮 Предсказания с лучшей моделью...")
            
            best_predictions = self._predict_with_model(best_model, prepared_molecules)
            self.predictions['egnn_model3'] = best_predictions
            
            logger.info(f"✅ Получено {len(best_predictions)} предсказаний")
            
            # 4. Ensemble предсказания для uncertainty estimation
            logger.info("\n🎯 Ensemble предсказания для uncertainty estimation...")
            
            ensemble_predictions = {}
            all_model_predictions = []
            
            for model_name, model_path in self.model_paths.items():
                if model_path.exists():
                    try:
                        model = self._load_egnn_model(model_path)
                        predictions = self._predict_with_model(model, prepared_molecules)
                        self.predictions[model_name] = predictions
                        all_model_predictions.append(predictions)
                        logger.info(f"✅ {model_name}: {len(predictions)} предсказаний")
                    except Exception as e:
                        logger.warning(f"Не удалось загрузить {model_name}: {e}")
            
            # Вычисляем ensemble статистики
            if len(all_model_predictions) >= 2:
                for mol_name in best_predictions.keys():
                    mol_predictions = [pred.get(mol_name) for pred in all_model_predictions if pred.get(mol_name) is not None]
                    
                    if len(mol_predictions) >= 2:
                        ensemble_predictions[mol_name] = {
                            'mean': np.mean(mol_predictions),
                            'std': np.std(mol_predictions),
                            'min': np.min(mol_predictions),
                            'max': np.max(mol_predictions),
                            'n_models': len(mol_predictions)
                        }
                
                self.ensemble_predictions = ensemble_predictions
                logger.info(f"✅ Ensemble статистики для {len(ensemble_predictions)} молекул")
            
            # 5. Сохранение результатов предсказаний
            logger.info("\n💾 Сохранение результатов предсказаний...")
            
            predictions_file = self.results_dir / "task_31_predictions.json"
            
            results = {
                'metadata': {
                    'timestamp': time.time(),
                    'best_model': 'egnn_model3',
                    'expected_qm9_performance': {
                        'mae': 0.076,
                        'r2': 0.9931
                    },
                    'n_molecules': len(prepared_molecules),
                    'n_ensemble_models': len(all_model_predictions)
                },
                'predictions': self.predictions,
                'ensemble_predictions': self.ensemble_predictions,
                'experimental_data': {mol['name']: {
                    'gap_energy': mol['gap_energy'],
                    'n_atoms': mol['n_atoms'],
                    'quality_score': mol['quality_score'],
                    'antibacterial_class': mol['antibacterial_class']
                } for mol in valid_molecules}
            }
            
            with open(predictions_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Результаты сохранены: {predictions_file}")
            
            # 6. Краткая сводка
            logger.info("\n✅ SUBTASK 31.2 ЗАВЕРШЕН")
            logger.info("="*60)
            logger.info(f"🎯 Лучшая модель: EGNN Model 3")
            logger.info(f"📊 Предсказаний получено: {len(best_predictions)}")
            logger.info(f"🎲 Ensemble моделей: {len(all_model_predictions)}")
            logger.info(f"📈 Uncertainty estimation: {'✅' if ensemble_predictions else '❌'}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка в Subtask 31.2: {e}")
            raise
    
    def run_subtask_31_3(self):
        """
        Subtask 31.3: Статистическое сравнение с экспериментальными данными
        """
        
        logger.info("🚀 SUBTASK 31.3: СТАТИСТИЧЕСКОЕ СРАВНЕНИЕ С ЭКСПЕРИМЕНТАЛЬНЫМИ ДАННЫМИ")
        logger.info("="*80)
        
        try:
            # Загружаем результаты предсказаний
            predictions_file = self.results_dir / "task_31_predictions.json"
            
            if not predictions_file.exists():
                raise FileNotFoundError("Сначала выполните Subtask 31.2")
            
            with open(predictions_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            best_predictions = results['predictions']['egnn_model3']
            experimental_data = results['experimental_data']
            
            # 1. Подготовка данных для анализа
            logger.info("\n📊 Подготовка данных для статистического анализа...")
            
            analysis_data = []
            
            for mol_name, pred_gap in best_predictions.items():
                if mol_name in experimental_data:
                    exp_data = experimental_data[mol_name]
                    exp_gap = exp_data['gap_energy']
                    
                    if exp_gap is not None:
                        analysis_data.append({
                            'name': mol_name,
                            'predicted_gap': pred_gap,
                            'experimental_gap': exp_gap,
                            'n_atoms': exp_data['n_atoms'],
                            'quality_score': exp_data['quality_score'],
                            'antibacterial_class': exp_data['antibacterial_class'],
                            'absolute_error': abs(pred_gap - exp_gap),
                            'relative_error': abs(pred_gap - exp_gap) / exp_gap * 100
                        })
            
            if not analysis_data:
                raise ValueError("Нет данных для статистического анализа")
            
            df = pd.DataFrame(analysis_data)
            logger.info(f"✅ Подготовлено {len(df)} пар для анализа")
            
            # 2. Основные метрики точности
            logger.info("\n📈 Вычисление основных метрик точности...")
            
            predicted = df['predicted_gap'].values
            experimental = df['experimental_gap'].values
            
            # MAE, RMSE, R²
            mae = np.mean(np.abs(predicted - experimental))
            rmse = np.sqrt(np.mean((predicted - experimental) ** 2))
            r2 = stats.pearsonr(predicted, experimental)[0] ** 2
            
            # Корреляции
            pearson_r, pearson_p = stats.pearsonr(predicted, experimental)
            spearman_r, spearman_p = stats.spearmanr(predicted, experimental)
            
            # Domain Shift Factor
            qm9_mae = 0.076  # Ожидаемая точность на QM9
            domain_shift_factor = mae / qm9_mae
            
            overall_metrics = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2,
                'pearson_correlation': pearson_r,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_r,
                'spearman_p_value': spearman_p,
                'domain_shift_factor': domain_shift_factor,
                'n_samples': len(df)
            }
            
            logger.info(f"📊 MAE: {mae:.3f} eV")
            logger.info(f"📊 RMSE: {rmse:.3f} eV")
            logger.info(f"📊 R²: {r2:.3f}")
            logger.info(f"📊 Pearson r: {pearson_r:.3f} (p={pearson_p:.3e})")
            logger.info(f"📊 Domain Shift Factor: {domain_shift_factor:.2f}x")
            
            # 3. Анализ по группам размеров
            logger.info("\n🔍 Анализ ошибок по группам размеров молекул...")
            
            # Определяем группы размеров
            def get_size_group(n_atoms):
                if n_atoms <= 30:
                    return 'small'
                elif n_atoms <= 60:
                    return 'medium'
                elif n_atoms <= 100:
                    return 'large'
                elif n_atoms <= 200:
                    return 'xlarge'
                else:
                    return 'xxlarge'
            
            df['size_group'] = df['n_atoms'].apply(get_size_group)
            
            size_group_metrics = {}
            
            for group in df['size_group'].unique():
                group_df = df[df['size_group'] == group]
                
                if len(group_df) >= 2:  # Минимум 2 точки для статистики
                    group_pred = group_df['predicted_gap'].values
                    group_exp = group_df['experimental_gap'].values
                    
                    group_mae = np.mean(np.abs(group_pred - group_exp))
                    group_rmse = np.sqrt(np.mean((group_pred - group_exp) ** 2))
                    
                    if len(group_df) >= 3:  # Минимум 3 точки для корреляции
                        group_r, group_p = stats.pearsonr(group_pred, group_exp)
                        group_r2 = group_r ** 2
                    else:
                        group_r, group_p, group_r2 = np.nan, np.nan, np.nan
                    
                    size_group_metrics[group] = {
                        'n_samples': len(group_df),
                        'mae': group_mae,
                        'rmse': group_rmse,
                        'r2': group_r2,
                        'pearson_r': group_r,
                        'pearson_p': group_p,
                        'domain_shift_factor': group_mae / qm9_mae,
                        'size_range': f"{group_df['n_atoms'].min()}-{group_df['n_atoms'].max()}"
                    }
                    
                    logger.info(f"  {group.upper()}: n={len(group_df)}, MAE={group_mae:.3f} eV, R²={group_r2:.3f}")
            
            # 4. Сохранение результатов анализа
            logger.info("\n💾 Сохранение результатов статистического анализа...")
            
            validation_results = {
                'metadata': {
                    'timestamp': time.time(),
                    'analysis_type': 'experimental_validation',
                    'qm9_baseline_mae': qm9_mae
                },
                'overall_metrics': overall_metrics,
                'size_group_metrics': size_group_metrics,
                'detailed_results': df.to_dict('records')
            }
            
            validation_file = self.results_dir / "task_31_validation_metrics.json"
            
            with open(validation_file, 'w', encoding='utf-8') as f:
                json.dump(validation_results, f, indent=2, ensure_ascii=False)
            
            # Сохраняем также CSV для удобства
            csv_file = self.results_dir / "task_31_validation_results.csv"
            df.to_csv(csv_file, index=False)
            
            logger.info(f"💾 Метрики сохранены: {validation_file}")
            logger.info(f"💾 Детальные результаты: {csv_file}")
            
            # 5. Краткая сводка
            logger.info("\n✅ SUBTASK 31.3 ЗАВЕРШЕН")
            logger.info("="*60)
            logger.info(f"📊 Общая точность: MAE={mae:.3f} eV, R²={r2:.3f}")
            logger.info(f"🔄 Domain Shift: {domain_shift_factor:.2f}x деградация от QM9")
            logger.info(f"📈 Статистическая значимость: p={pearson_p:.2e}")
            logger.info(f"🎯 Групп размеров проанализировано: {len(size_group_metrics)}")
            
            self.validation_metrics = validation_results
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка в Subtask 31.3: {e}")
            raise
    
    def run_full_task_31(self):
        """Запускает полную Task 31."""
        
        logger.info("🚀 ЗАПУСК ПОЛНОЙ TASK 31: ВАЛИДАЦИЯ НА ЭКСПЕРИМЕНТАЛЬНЫХ ДАННЫХ")
        logger.info("="*80)
        
        try:
            # Subtask 31.1 уже выполнен (расширенный поиск данных)
            logger.info("✅ Subtask 31.1: Поиск экспериментальных данных - ЗАВЕРШЕН")
            
            # Subtask 31.2: Предсказания
            self.run_subtask_31_2()
            
            # Subtask 31.3: Статистическое сравнение
            self.run_subtask_31_3()
            
            # TODO: Subtask 31.4: Визуализации и отчет
            # TODO: Subtask 31.5: Интеграция с существующими результатами
            
            logger.info("\n🎉 TASK 31 ЧАСТИЧНО ЗАВЕРШЕНА")
            logger.info("✅ Subtasks 31.1-31.3 выполнены")
            logger.info("🔄 Subtasks 31.4-31.5 требуют дополнительной реализации")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Ошибка в Task 31: {e}")
            raise


def main():
    """Главная функция."""
    
    try:
        validator = Task31ExperimentalValidator()
        validator.run_full_task_31()
        
    except Exception as e:
        logger.error(f"❌ Ошибка в main: {e}")
        raise


if __name__ == "__main__":
    main()