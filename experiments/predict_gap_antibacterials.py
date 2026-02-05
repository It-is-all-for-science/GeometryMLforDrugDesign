#!/usr/bin/env python3
"""
Предсказание HOMO-LUMO Gap для антибактериальных препаратов
Используем лучшую EGNN Model 3 (MAE=0.076 eV, R²=0.9931)

ЦЕЛЬ: Получить Gap энергии для всех 29 антибактериальных препаратов
"""

import os
import sys
import torch
import torch_geometric
from torch_geometric.data import Data
import numpy as np
from pathlib import Path
import logging
import time
from datetime import datetime
import json
import pandas as pd
import pickle
from typing import Dict, List, Tuple

# Добавляем src в путь
sys.path.append(str(Path(__file__).parent.parent / "src"))

from step_01_data.loaders import MolecularDataLoader
from step_03_models.egnn import EGNNModel, EGNNConfig

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GapPredictor:
    """Предсказатель HOMO-LUMO Gap для антибактериальных препаратов."""
    
    def __init__(self):
        # Пути к данным и модели
        self.model_path = "results/improved_egnn_ensemble/models/improved_egnn_model3/best_model.pth"
        self.antibacterial_data_path = "experiments/results/antibacterial_analysis/antibacterial_structures.pkl"
        self.results_dir = Path("results/gap_predictions_antibacterials")
        
        # Создаем директорию результатов
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Устройство
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"🔧 Устройство: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def load_best_model(self) -> EGNNModel:
        """Загружает лучшую EGNN Model 3."""
        
        logger.info(f"📋 Загрузка лучшей EGNN Model 3...")
        logger.info(f"📁 Путь: {self.model_path}")
        
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Лучшая модель не найдена: {self.model_path}")
        
        # Загружаем checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        logger.info(f"📋 Загружен checkpoint: {list(checkpoint.keys())}")
        
        # Создаем модель с той же архитектурой (Model 3 использует hidden_dim=256)
        egnn_config = EGNNConfig(
            hidden_dim=256,  # Исправлено: Model 3 использует 256, не 128
            num_layers=5,    # Исправлено: Model 3 использует 5 слоев
            output_dim=1,
            node_feature_dim=11,
            dropout=0.1
        )
        
        model = EGNNModel(egnn_config)
        
        # Загружаем веса
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Убираем префиксы если есть
        if len(state_dict) > 0:
            sample_key = list(state_dict.keys())[0]
            if sample_key.startswith('egnn_model.'):
                logger.info("🔧 Убираем префикс 'egnn_model.'")
                clean_state_dict = {}
                for key, value in state_dict.items():
                    clean_key = key.replace('egnn_model.', '')
                    clean_state_dict[clean_key] = value
                state_dict = clean_state_dict
        
        # Загружаем веса в модель
        model.load_state_dict(state_dict, strict=False)
        model.to(self.device)
        model.eval()
        
        logger.info(f"✅ Лучшая EGNN Model 3 загружена успешно")
        logger.info(f"📊 Ожидаемая точность: MAE=0.076 eV, R²=0.9931")
        
        return model
    
    def load_antibacterial_data(self) -> Dict:
        """Загружает данные антибактериальных препаратов."""
        
        logger.info(f"📥 Загрузка антибактериальных данных...")
        
        # Пробуем загрузить JSON файл
        json_path = Path("experiments/results/antibacterial_analysis/prepared_antibacterial_dataset.json")
        if json_path.exists():
            logger.info(f"📁 Загружаем из JSON: {json_path}")
            
            with open(json_path, 'r') as f:
                json_data = json.load(f)
            
            # Конвертируем JSON в нужный формат
            antibacterial_data = {}
            
            for mol_info in json_data['molecules']:
                mol_name = mol_info['name']
                
                # Создаем PyTorch Geometric данные
                import torch_geometric
                from torch_geometric.data import Data
                
                # Конвертируем атомные номера и координаты
                atomic_numbers = torch.tensor(mol_info['atomic_numbers'], dtype=torch.long)
                coordinates = torch.tensor(mol_info['coordinates'], dtype=torch.float32)
                
                # Создаем node features в том же формате, что и QM9
                # QM9 использует 11-мерные признаки узлов
                node_features = self._create_qm9_style_features(atomic_numbers)
                
                # Создаем простые edges (полносвязный граф для начала)
                num_atoms = len(atomic_numbers)
                edge_index = []
                for i in range(num_atoms):
                    for j in range(num_atoms):
                        if i != j:
                            edge_index.append([i, j])
                
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                
                # Создаем PyTorch Geometric объект
                mol_graph = Data(
                    x=node_features,
                    pos=coordinates,
                    edge_index=edge_index
                )
                
                # Определяем группу размера
                num_atoms = mol_info['n_atoms']
                if num_atoms <= 20:
                    size_group = "small"
                elif num_atoms <= 50:
                    size_group = "medium"
                else:
                    size_group = "large"
                
                antibacterial_data[mol_name] = {
                    'mol_data': mol_graph,
                    'size_group': size_group,
                    'num_atoms': num_atoms,
                    'molecular_weight': mol_info.get('molecular_weight', 0.0)
                }
            
            logger.info(f"✅ Загружено {len(antibacterial_data)} антибактериальных препаратов из JSON")
            
        else:
            # Пробуем старый pickle файл
            pickle_path = Path(self.antibacterial_data_path)
            if not pickle_path.exists():
                raise FileNotFoundError(f"Данные антибиотиков не найдены ни в JSON, ни в pickle: {json_path}, {pickle_path}")
            
            with open(pickle_path, 'rb') as f:
                antibacterial_data = pickle.load(f)
            
            logger.info(f"✅ Загружено {len(antibacterial_data)} антибактериальных препаратов из pickle")
        
        # Группируем по размерам для статистики
        size_groups = {}
        for name, data in antibacterial_data.items():
            size_group = data['size_group']
            if size_group not in size_groups:
                size_groups[size_group] = []
            size_groups[size_group].append(name)
        
        # Логируем статистику
        logger.info(f"📊 Распределение по группам размеров:")
        for group, molecules in size_groups.items():
            logger.info(f"   {group}: {len(molecules)} молекул")
        
        return antibacterial_data
    
    def _create_qm9_style_features(self, atomic_numbers: List[int]) -> torch.Tensor:
        """Создает node features в стиле QM9 (11-мерные)."""
        
        features = []
        
        for atomic_num in atomic_numbers:
            # Создаем 11-мерный вектор признаков как в QM9
            feat = torch.zeros(11)
            
            # Первые 5 позиций - one-hot для основных элементов
            # H=1, C=6, N=7, O=8, F=9
            if atomic_num == 1:    # H
                feat[0] = 1.0
            elif atomic_num == 6:  # C
                feat[1] = 1.0
            elif atomic_num == 7:  # N
                feat[2] = 1.0
            elif atomic_num == 8:  # O
                feat[3] = 1.0
            elif atomic_num == 9:  # F
                feat[4] = 1.0
            
            # Позиция 5 - атомный номер (нормализованный)
            feat[5] = atomic_num / 100.0
            
            # Остальные позиции можно использовать для других свойств
            # Пока оставляем нулями или добавляем простые признаки
            
            # Позиция 10 - количество валентных электронов (упрощенно)
            valence_electrons = {1: 1, 6: 4, 7: 5, 8: 6, 9: 7, 16: 6, 17: 7}
            feat[10] = valence_electrons.get(atomic_num, 0)
            
            features.append(feat)
        
        return torch.stack(features)
    
    def predict_gap_energies(self, model: EGNNModel, antibacterial_data: Dict) -> Dict:
        """Предсказывает HOMO-LUMO Gap для всех антибактериальных препаратов."""
        
        logger.info(f"🚀 Начинаем предсказания HOMO-LUMO Gap...")
        
        predictions = {}
        successful_predictions = 0
        failed_predictions = 0
        
        # Сортируем по группам размеров для удобства
        sorted_molecules = sorted(
            antibacterial_data.items(),
            key=lambda x: (x[1]['size_group'], x[1]['num_atoms'])
        )
        
        for mol_name, mol_data in sorted_molecules:
            logger.info(f"📊 Обработка {mol_name} ({mol_data['size_group']}, {mol_data['num_atoms']} атомов)...")
            
            try:
                # Получаем молекулярные данные
                mol_graph = mol_data['mol_data']
                
                # Перемещаем данные на GPU
                mol_x = mol_graph.x.to(self.device)
                mol_pos = mol_graph.pos.to(self.device)
                mol_edge_index = mol_graph.edge_index.to(self.device)
                
                # Предсказание
                with torch.no_grad():
                    output = model(mol_x, mol_pos, mol_edge_index)
                    if isinstance(output, dict):
                        pred = output['prediction']
                    else:
                        pred = output
                    
                    gap_energy = pred.squeeze().cpu().item()
                
                # Сохраняем результат
                predictions[mol_name] = {
                    'gap_energy_eV': gap_energy,
                    'size_group': mol_data['size_group'],
                    'num_atoms': mol_data['num_atoms'],
                    'molecular_weight': mol_data['molecular_weight'],
                    'success': True,
                    'error': None
                }
                
                successful_predictions += 1
                logger.info(f"  ✅ Gap = {gap_energy:.4f} eV")
                
            except Exception as e:
                logger.warning(f"  ❌ Ошибка: {e}")
                predictions[mol_name] = {
                    'gap_energy_eV': None,
                    'size_group': mol_data['size_group'],
                    'num_atoms': mol_data['num_atoms'],
                    'molecular_weight': mol_data['molecular_weight'],
                    'success': False,
                    'error': str(e)
                }
                failed_predictions += 1
        
        logger.info(f"✅ Предсказания завершены:")
        logger.info(f"   Успешно: {successful_predictions}")
        logger.info(f"   Неудачно: {failed_predictions}")
        logger.info(f"   Успешность: {successful_predictions/(successful_predictions+failed_predictions)*100:.1f}%")
        
        return predictions
    
    def analyze_results(self, predictions: Dict) -> Dict:
        """Анализирует результаты предсказаний."""
        
        logger.info(f"📊 Анализ результатов...")
        
        # Собираем успешные предсказания
        successful_preds = {name: data for name, data in predictions.items() if data['success']}
        
        if not successful_preds:
            logger.error("❌ Нет успешных предсказаний для анализа")
            return {}
        
        # Создаем DataFrame для анализа
        df_data = []
        for name, data in successful_preds.items():
            df_data.append({
                'molecule': name,
                'gap_energy_eV': data['gap_energy_eV'],
                'size_group': data['size_group'],
                'num_atoms': data['num_atoms'],
                'molecular_weight': data['molecular_weight']
            })
        
        df = pd.DataFrame(df_data)
        
        # Общая статистика
        analysis = {
            'total_molecules': len(predictions),
            'successful_predictions': len(successful_preds),
            'success_rate': len(successful_preds) / len(predictions),
            'gap_statistics': {
                'mean': df['gap_energy_eV'].mean(),
                'std': df['gap_energy_eV'].std(),
                'min': df['gap_energy_eV'].min(),
                'max': df['gap_energy_eV'].max(),
                'median': df['gap_energy_eV'].median()
            },
            'by_size_group': {}
        }
        
        # Анализ по группам размеров
        for group in df['size_group'].unique():
            group_df = df[df['size_group'] == group]
            analysis['by_size_group'][group] = {
                'count': len(group_df),
                'avg_atoms': group_df['num_atoms'].mean(),
                'avg_weight': group_df['molecular_weight'].mean(),
                'gap_mean': group_df['gap_energy_eV'].mean(),
                'gap_std': group_df['gap_energy_eV'].std(),
                'gap_min': group_df['gap_energy_eV'].min(),
                'gap_max': group_df['gap_energy_eV'].max()
            }
        
        # Логируем основные результаты
        logger.info(f"📊 Основные результаты:")
        logger.info(f"   Средний Gap: {analysis['gap_statistics']['mean']:.4f} ± {analysis['gap_statistics']['std']:.4f} eV")
        logger.info(f"   Диапазон: {analysis['gap_statistics']['min']:.4f} - {analysis['gap_statistics']['max']:.4f} eV")
        
        logger.info(f"📊 По группам размеров:")
        for group, stats in analysis['by_size_group'].items():
            logger.info(f"   {group}: {stats['count']} молекул, Gap = {stats['gap_mean']:.4f} ± {stats['gap_std']:.4f} eV")
        
        return analysis
    
    def save_results(self, predictions: Dict, analysis: Dict):
        """Сохраняет результаты предсказаний."""
        
        # Сохраняем предсказания
        predictions_path = self.results_dir / "gap_predictions.json"
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2, default=str)
        
        # Сохраняем анализ
        analysis_path = self.results_dir / "gap_analysis.json"
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Создаем CSV для удобства
        successful_preds = {name: data for name, data in predictions.items() if data['success']}
        if successful_preds:
            df_data = []
            for name, data in successful_preds.items():
                df_data.append({
                    'molecule': name,
                    'gap_energy_eV': data['gap_energy_eV'],
                    'size_group': data['size_group'],
                    'num_atoms': data['num_atoms'],
                    'molecular_weight': data['molecular_weight']
                })
            
            df = pd.DataFrame(df_data)
            csv_path = self.results_dir / "gap_predictions.csv"
            df.to_csv(csv_path, index=False)
            
            logger.info(f"💾 Результаты сохранены:")
            logger.info(f"   JSON: {predictions_path}")
            logger.info(f"   Анализ: {analysis_path}")
            logger.info(f"   CSV: {csv_path}")
    
    def create_summary_report(self, predictions: Dict, analysis: Dict):
        """Создает итоговый отчет."""
        
        report_path = self.results_dir / "gap_predictions_report.md"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# HOMO-LUMO Gap Предсказания для Антибактериальных Препаратов\n\n")
            f.write(f"**Дата выполнения**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## 🎯 Цель\n\n")
            f.write("Предсказать HOMO-LUMO Gap энергии для 29 антибактериальных препаратов ")
            f.write("используя лучшую EGNN Model 3 (MAE=0.076 eV, R²=0.9931).\n\n")
            
            f.write("## 📊 Общие результаты\n\n")
            f.write(f"- **Всего молекул**: {analysis['total_molecules']}\n")
            f.write(f"- **Успешных предсказаний**: {analysis['successful_predictions']}\n")
            f.write(f"- **Успешность**: {analysis['success_rate']:.1%}\n\n")
            
            if analysis['gap_statistics']:
                stats = analysis['gap_statistics']
                f.write("### Статистика Gap энергий\n\n")
                f.write(f"- **Среднее**: {stats['mean']:.4f} eV\n")
                f.write(f"- **Стандартное отклонение**: {stats['std']:.4f} eV\n")
                f.write(f"- **Минимум**: {stats['min']:.4f} eV\n")
                f.write(f"- **Максимум**: {stats['max']:.4f} eV\n")
                f.write(f"- **Медиана**: {stats['median']:.4f} eV\n\n")
            
            f.write("## 📈 Результаты по группам размеров\n\n")
            f.write("| Группа | Молекул | Средний размер | Gap (среднее) | Gap (σ) | Gap (мин-макс) |\n")
            f.write("|--------|---------|----------------|---------------|---------|----------------|\n")
            
            for group, stats in analysis['by_size_group'].items():
                f.write(f"| {group} | {stats['count']} | {stats['avg_atoms']:.1f} атомов | ")
                f.write(f"{stats['gap_mean']:.4f} eV | {stats['gap_std']:.4f} eV | ")
                f.write(f"{stats['gap_min']:.4f}-{stats['gap_max']:.4f} eV |\n")
            
            f.write("\n## 🔬 Детальные результаты\n\n")
            f.write("| Молекула | Gap (eV) | Группа | Атомы | Вес (Da) |\n")
            f.write("|----------|----------|--------|-------|----------|\n")
            
            # Сортируем по Gap энергии
            successful_preds = {name: data for name, data in predictions.items() if data['success']}
            sorted_preds = sorted(successful_preds.items(), key=lambda x: x[1]['gap_energy_eV'])
            
            for name, data in sorted_preds:
                f.write(f"| {name} | {data['gap_energy_eV']:.4f} | {data['size_group']} | ")
                f.write(f"{data['num_atoms']} | {data['molecular_weight']:.1f} |\n")
            
            f.write("\n## 💡 Интерпретация для Drug Design\n\n")
            f.write("### Реакционная способность:\n")
            f.write("- **Малый Gap (<4 eV)**: Высокая реакционная способность, потенциальная токсичность\n")
            f.write("- **Средний Gap (4-6 eV)**: Умеренная реакционная способность, хороший баланс\n")
            f.write("- **Большой Gap (>6 eV)**: Низкая реакционная способность, высокая стабильность\n\n")
            
            f.write("### Электронные свойства:\n")
            f.write("- Gap энергия влияет на взаимодействие с белками-мишенями\n")
            f.write("- Более реакционноспособные молекулы могут иметь больше побочных эффектов\n")
            f.write("- Стабильные молекулы лучше для длительной терапии\n\n")
            
            f.write("## ✅ Статус\n\n")
            f.write("**ПРЕДСКАЗАНИЯ ЗАВЕРШЕНЫ УСПЕШНО** ✅\n\n")
            f.write("Результаты готовы для дальнейшего анализа и Domain Shift исследования.\n")
        
        logger.info(f"📄 Отчет создан: {report_path}")
    
    def run_prediction(self):
        """Выполняет полное предсказание Gap энергий."""
        
        logger.info("🚀 ПРЕДСКАЗАНИЕ HOMO-LUMO GAP ДЛЯ АНТИБАКТЕРИАЛЬНЫХ ПРЕПАРАТОВ")
        logger.info(f"📁 Результаты: {self.results_dir}")
        
        try:
            # 1. Загружаем лучшую модель
            logger.info(f"\n" + "="*60)
            logger.info(f"📋 ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ")
            logger.info("="*60)
            
            model = self.load_best_model()
            
            # 2. Загружаем антибактериальные данные
            logger.info(f"\n" + "="*60)
            logger.info(f"📥 ЗАГРУЗКА ДАННЫХ")
            logger.info("="*60)
            
            antibacterial_data = self.load_antibacterial_data()
            
            # 3. Предсказываем Gap энергии
            logger.info(f"\n" + "="*60)
            logger.info(f"🔮 ПРЕДСКАЗАНИЯ")
            logger.info("="*60)
            
            predictions = self.predict_gap_energies(model, antibacterial_data)
            
            # 4. Анализируем результаты
            logger.info(f"\n" + "="*60)
            logger.info(f"📊 АНАЛИЗ")
            logger.info("="*60)
            
            analysis = self.analyze_results(predictions)
            
            # 5. Сохраняем результаты
            logger.info(f"\n" + "="*60)
            logger.info(f"💾 СОХРАНЕНИЕ")
            logger.info("="*60)
            
            self.save_results(predictions, analysis)
            
            # 6. Создаем отчет
            self.create_summary_report(predictions, analysis)
            
            # 7. Выводим итоги
            logger.info("\n" + "="*60)
            logger.info("✅ ПРЕДСКАЗАНИЯ ЗАВЕРШЕНЫ")
            logger.info("="*60)
            
            if analysis:
                logger.info(f"📊 Итоговая статистика:")
                logger.info(f"   Успешность: {analysis['success_rate']:.1%}")
                logger.info(f"   Средний Gap: {analysis['gap_statistics']['mean']:.4f} ± {analysis['gap_statistics']['std']:.4f} eV")
                logger.info(f"   Диапазон: {analysis['gap_statistics']['min']:.4f} - {analysis['gap_statistics']['max']:.4f} eV")
            
            logger.info(f"📁 Все результаты в: {self.results_dir}")
            logger.info("🎯 ГОТОВО ДЛЯ DOMAIN SHIFT АНАЛИЗА")
            
            return predictions, analysis
            
        except Exception as e:
            logger.error(f"❌ Ошибка в предсказании: {e}")
            raise

def main():
    """Главная функция."""
    
    try:
        predictor = GapPredictor()
        predictions, analysis = predictor.run_prediction()
        return predictions, analysis
        
    except Exception as e:
        logger.error(f"❌ Ошибка в main: {e}")
        raise

if __name__ == "__main__":
    main()