#!/usr/bin/env python3
"""
Оптимизированный анализ антибактериальных препаратов с правильной группировкой по размерам.
Использует найденные структуры и перегруппировывает их по реальным размерам.
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import pickle

# Добавляем путь к нашим модулям
sys.path.append(str(Path(__file__).parent.parent / "src"))

# Импорты будут добавлены позже при необходимости
# from step_01_data_loading.molecular_data_loader import MolecularDataLoader
# from step_04_egnn.egnn_model import EGNNModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedAntibacterialAnalysis:
    """
    Оптимизированный анализ антибактериальных препаратов.
    
    Использует найденные структуры и правильно группирует их по размерам.
    Обеспечивает по 10 молекул в каждой группе (кроме очень больших).
    """
    
    def __init__(self, cache_dir: str = "data/antibacterial_cache"):
        self.cache_dir = Path(cache_dir)
        self.results_dir = Path("results/antibacterial_analysis")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Загружаем найденные структуры
        self.structures_file = self.cache_dir / "antibacterial_structures_for_analysis.json"
        self.structures_data = self._load_structures()
        
        # Определяем оптимальные группы размеров на основе реальных данных
        self.optimized_groups = self._create_optimized_groups()
        
        # Инициализируем загрузчик данных и модель
        self.data_loader = None
        self.model = None
        
    def _load_structures(self) -> Dict:
        """Загружает найденные структуры."""
        
        if not self.structures_file.exists():
            logger.error(f"❌ Файл структур не найден: {self.structures_file}")
            logger.info("🔧 Запустите сначала reliable_antibacterial_structure_finder.py")
            raise FileNotFoundError(f"Structures file not found: {self.structures_file}")
        
        with open(self.structures_file, 'r') as f:
            data = json.load(f)
        
        logger.info(f"📋 Загружены структуры: {data['metadata']['total_structures']} молекул")
        return data
    
    def _create_optimized_groups(self) -> Dict:
        """Создает оптимальные группы размеров на основе реальных данных."""
        
        # Собираем все молекулы и их размеры
        all_molecules = []
        
        for group_name, molecules in self.structures_data['structures'].items():
            for molecule in molecules:
                all_molecules.append({
                    'name': molecule['name'],
                    'n_atoms': molecule['n_atoms'],
                    'original_group': group_name,
                    'antibacterial_class': molecule.get('antibacterial_class', 'unknown'),
                    'mechanism_of_action': molecule.get('mechanism_of_action', 'unknown'),
                    'data': molecule
                })
        
        # Сортируем по размеру
        all_molecules.sort(key=lambda x: x['n_atoms'])
        
        logger.info(f"📊 Анализ размеров {len(all_molecules)} молекул:")
        sizes = [mol['n_atoms'] for mol in all_molecules]
        logger.info(f"   Минимум: {min(sizes)} атомов")
        logger.info(f"   Максимум: {max(sizes)} атомов")
        logger.info(f"   Медиана: {np.median(sizes):.1f} атомов")
        logger.info(f"   Среднее: {np.mean(sizes):.1f} атомов")
        
        # Создаем оптимальные группы
        optimized_groups = {
            'tiny': {
                'size_range': (10, 20),
                'target_count': 10,
                'description': 'Очень малые антибиотики (10-20 атомов)',
                'molecules': []
            },
            'small': {
                'size_range': (21, 30),
                'target_count': 10,
                'description': 'Малые антибиотики (21-30 атомов)',
                'molecules': []
            },
            'medium': {
                'size_range': (31, 50),
                'target_count': 10,
                'description': 'Средние антибиотики (31-50 атомов)',
                'molecules': []
            },
            'large': {
                'size_range': (51, 80),
                'target_count': 8,
                'description': 'Большие антибиотики (51-80 атомов)',
                'molecules': []
            },
            'xlarge': {
                'size_range': (81, 150),
                'target_count': 5,
                'description': 'Очень большие антибиотики (81-150 атомов)',
                'molecules': []
            },
            'xxlarge': {
                'size_range': (151, 300),
                'target_count': 3,
                'description': 'Гигантские антибиотики (151-300 атомов)',
                'molecules': []
            }
        }
        
        # Распределяем молекулы по группам
        for molecule in all_molecules:
            n_atoms = molecule['n_atoms']
            
            for group_name, group_config in optimized_groups.items():
                min_size, max_size = group_config['size_range']
                
                if min_size <= n_atoms <= max_size:
                    if len(group_config['molecules']) < group_config['target_count']:
                        group_config['molecules'].append(molecule)
                    break
        
        # Выводим статистику по группам
        logger.info(f"\n📈 ОПТИМИЗИРОВАННЫЕ ГРУППЫ:")
        
        for group_name, group_config in optimized_groups.items():
            molecules = group_config['molecules']
            target_count = group_config['target_count']
            size_range = group_config['size_range']
            description = group_config['description']
            
            status = "✅" if len(molecules) >= target_count else "⚠️"
            
            logger.info(f"  {status} {group_name.upper()}: {len(molecules)}/{target_count}")
            logger.info(f"      {description}")
            
            if molecules:
                sizes = [mol['n_atoms'] for mol in molecules]
                logger.info(f"      Реальные размеры: {min(sizes)}-{max(sizes)} атомов")
                
                # Примеры
                for i, mol in enumerate(molecules[:3], 1):
                    logger.info(f"        {i}. {mol['name']}: {mol['n_atoms']} атомов")
        
        return optimized_groups
    
    def prepare_analysis_dataset(self) -> Dict:
        """Подготавливает датасет для анализа."""
        
        logger.info("🔧 Подготовка датасета для анализа...")
        
        analysis_dataset = {
            'molecules': [],
            'groups': {},
            'metadata': {
                'total_molecules': 0,
                'groups_count': len(self.optimized_groups),
                'preparation_timestamp': time.time()
            }
        }
        
        for group_name, group_config in self.optimized_groups.items():
            molecules = group_config['molecules']
            
            if not molecules:
                logger.warning(f"⚠️ Группа {group_name} пуста")
                continue
            
            group_data = {
                'name': group_name,
                'description': group_config['description'],
                'size_range': group_config['size_range'],
                'molecules': [],
                'statistics': {}
            }
            
            for molecule in molecules:
                mol_data = molecule['data']
                
                # Подготавливаем данные для ML анализа
                prepared_molecule = {
                    'id': mol_data['id'],
                    'name': mol_data['name'],
                    'n_atoms': mol_data['n_atoms'],
                    'atomic_numbers': mol_data['atomic_numbers'],
                    'coordinates': mol_data['coordinates'],
                    'smiles': mol_data['smiles'],
                    'molecular_weight': mol_data.get('molecular_weight', 0),
                    'logp': mol_data.get('logp', 0),
                    'tpsa': mol_data.get('tpsa', 0),
                    'antibacterial_class': mol_data.get('antibacterial_class', 'unknown'),
                    'mechanism_of_action': mol_data.get('mechanism_of_action', 'unknown'),
                    'group': group_name,
                    'quality_score': mol_data.get('quality_score', 0.8)
                }
                
                group_data['molecules'].append(prepared_molecule)
                analysis_dataset['molecules'].append(prepared_molecule)
            
            # Вычисляем статистику группы
            group_molecules = group_data['molecules']
            
            if group_molecules:
                sizes = [mol['n_atoms'] for mol in group_molecules]
                weights = [mol['molecular_weight'] for mol in group_molecules]
                logps = [mol['logp'] for mol in group_molecules]
                
                group_data['statistics'] = {
                    'count': len(group_molecules),
                    'size_stats': {
                        'min': min(sizes),
                        'max': max(sizes),
                        'mean': np.mean(sizes),
                        'median': np.median(sizes),
                        'std': np.std(sizes)
                    },
                    'weight_stats': {
                        'min': min(weights),
                        'max': max(weights),
                        'mean': np.mean(weights),
                        'median': np.median(weights)
                    },
                    'logp_stats': {
                        'min': min(logps),
                        'max': max(logps),
                        'mean': np.mean(logps),
                        'median': np.median(logps)
                    }
                }
            
            analysis_dataset['groups'][group_name] = group_data
        
        analysis_dataset['metadata']['total_molecules'] = len(analysis_dataset['molecules'])
        
        # Сохраняем подготовленный датасет
        dataset_file = self.results_dir / "prepared_antibacterial_dataset.json"
        with open(dataset_file, 'w') as f:
            json.dump(analysis_dataset, f, indent=2)
        
        logger.info(f"✅ Подготовлен датасет: {analysis_dataset['metadata']['total_molecules']} молекул")
        logger.info(f"📁 Сохранен в: {dataset_file}")
        
        return analysis_dataset
    
    def create_visualization_summary(self, analysis_dataset: Dict):
        """Создает визуализацию сводки по датасету."""
        
        logger.info("📊 Создание визуализации сводки...")
        
        # Настройка стиля
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Сводка по антибактериальным препаратам', fontsize=16, fontweight='bold')
        
        # 1. Распределение по группам размеров
        ax1 = axes[0, 0]
        groups = list(analysis_dataset['groups'].keys())
        counts = [len(analysis_dataset['groups'][group]['molecules']) for group in groups]
        
        bars = ax1.bar(groups, counts, alpha=0.7)
        ax1.set_title('Количество молекул по группам')
        ax1.set_ylabel('Количество молекул')
        ax1.tick_params(axis='x', rotation=45)
        
        # Добавляем значения на столбцы
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{count}', ha='center', va='bottom')
        
        # 2. Распределение размеров молекул
        ax2 = axes[0, 1]
        all_sizes = [mol['n_atoms'] for mol in analysis_dataset['molecules']]
        
        ax2.hist(all_sizes, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Распределение размеров молекул')
        ax2.set_xlabel('Количество атомов')
        ax2.set_ylabel('Частота')
        ax2.axvline(np.mean(all_sizes), color='red', linestyle='--', 
                   label=f'Среднее: {np.mean(all_sizes):.1f}')
        ax2.legend()
        
        # 3. Распределение по классам антибиотиков
        ax3 = axes[0, 2]
        classes = [mol['antibacterial_class'] for mol in analysis_dataset['molecules']]
        class_counts = pd.Series(classes).value_counts()
        
        wedges, texts, autotexts = ax3.pie(class_counts.values, labels=class_counts.index, 
                                          autopct='%1.1f%%', startangle=90)
        ax3.set_title('Распределение по классам антибиотиков')
        
        # 4. Распределение по механизмам действия
        ax4 = axes[1, 0]
        mechanisms = [mol['mechanism_of_action'] for mol in analysis_dataset['molecules']]
        mechanism_counts = pd.Series(mechanisms).value_counts()
        
        ax4.barh(range(len(mechanism_counts)), mechanism_counts.values)
        ax4.set_yticks(range(len(mechanism_counts)))
        ax4.set_yticklabels([mech.replace('_', ' ').title() for mech in mechanism_counts.index])
        ax4.set_title('Распределение по механизмам действия')
        ax4.set_xlabel('Количество молекул')
        
        # 5. Молекулярный вес vs LogP
        ax5 = axes[1, 1]
        weights = [mol['molecular_weight'] for mol in analysis_dataset['molecules']]
        logps = [mol['logp'] for mol in analysis_dataset['molecules']]
        groups_for_color = [mol['group'] for mol in analysis_dataset['molecules']]
        
        scatter = ax5.scatter(weights, logps, c=range(len(weights)), 
                             cmap='viridis', alpha=0.7)
        ax5.set_xlabel('Молекулярный вес (Da)')
        ax5.set_ylabel('LogP')
        ax5.set_title('Молекулярный вес vs Липофильность')
        
        # 6. Статистика по группам
        ax6 = axes[1, 2]
        group_names = []
        mean_sizes = []
        std_sizes = []
        
        for group_name, group_data in analysis_dataset['groups'].items():
            if group_data['molecules']:
                group_names.append(group_name)
                stats = group_data['statistics']['size_stats']
                mean_sizes.append(stats['mean'])
                std_sizes.append(stats['std'])
        
        x_pos = range(len(group_names))
        ax6.bar(x_pos, mean_sizes, yerr=std_sizes, alpha=0.7, capsize=5)
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(group_names, rotation=45)
        ax6.set_title('Средний размер молекул по группам')
        ax6.set_ylabel('Количество атомов')
        
        plt.tight_layout()
        
        # Сохраняем график
        plot_file = self.results_dir / "antibacterial_summary_visualization.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Визуализация сохранена: {plot_file}")
    
    def create_detailed_report(self, analysis_dataset: Dict) -> str:
        """Создает детальный отчет по анализу."""
        
        logger.info("📝 Создание детального отчета...")
        
        report_lines = []
        report_lines.append("# Анализ антибактериальных препаратов")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Общая информация
        total_molecules = analysis_dataset['metadata']['total_molecules']
        groups_count = analysis_dataset['metadata']['groups_count']
        
        report_lines.append(f"## Общая информация")
        report_lines.append(f"- **Всего молекул**: {total_molecules}")
        report_lines.append(f"- **Количество групп**: {groups_count}")
        report_lines.append(f"- **Дата анализа**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Статистика по группам
        report_lines.append("## Статистика по группам размеров")
        report_lines.append("")
        
        for group_name, group_data in analysis_dataset['groups'].items():
            molecules = group_data['molecules']
            
            if not molecules:
                continue
            
            stats = group_data['statistics']
            size_stats = stats['size_stats']
            weight_stats = stats['weight_stats']
            logp_stats = stats['logp_stats']
            
            report_lines.append(f"### {group_name.upper()}: {group_data['description']}")
            report_lines.append(f"- **Количество молекул**: {stats['count']}")
            report_lines.append(f"- **Размер атомов**: {size_stats['min']}-{size_stats['max']} "
                               f"(среднее: {size_stats['mean']:.1f} ± {size_stats['std']:.1f})")
            report_lines.append(f"- **Молекулярный вес**: {weight_stats['min']:.1f}-{weight_stats['max']:.1f} Da "
                               f"(среднее: {weight_stats['mean']:.1f})")
            report_lines.append(f"- **LogP**: {logp_stats['min']:.2f}-{logp_stats['max']:.2f} "
                               f"(среднее: {logp_stats['mean']:.2f})")
            report_lines.append("")
            
            # Список молекул в группе
            report_lines.append("**Молекулы в группе:**")
            for i, mol in enumerate(molecules, 1):
                report_lines.append(f"{i}. **{mol['name']}** ({mol['n_atoms']} атомов)")
                report_lines.append(f"   - Класс: {mol['antibacterial_class'].replace('_', ' ').title()}")
                report_lines.append(f"   - Механизм: {mol['mechanism_of_action'].replace('_', ' ').title()}")
                report_lines.append(f"   - Молекулярный вес: {mol['molecular_weight']:.1f} Da")
                report_lines.append(f"   - LogP: {mol['logp']:.2f}")
            report_lines.append("")
        
        # Анализ по классам
        report_lines.append("## Анализ по классам антибиотиков")
        report_lines.append("")
        
        classes = [mol['antibacterial_class'] for mol in analysis_dataset['molecules']]
        class_counts = pd.Series(classes).value_counts()
        
        for class_name, count in class_counts.items():
            percentage = (count / total_molecules) * 100
            report_lines.append(f"- **{class_name.replace('_', ' ').title()}**: {count} молекул ({percentage:.1f}%)")
        
        report_lines.append("")
        
        # Анализ по механизмам действия
        report_lines.append("## Анализ по механизмам действия")
        report_lines.append("")
        
        mechanisms = [mol['mechanism_of_action'] for mol in analysis_dataset['molecules']]
        mechanism_counts = pd.Series(mechanisms).value_counts()
        
        for mechanism, count in mechanism_counts.items():
            percentage = (count / total_molecules) * 100
            report_lines.append(f"- **{mechanism.replace('_', ' ').title()}**: {count} молекул ({percentage:.1f}%)")
        
        report_lines.append("")
        
        # Рекомендации для ML анализа
        report_lines.append("## Рекомендации для ML анализа")
        report_lines.append("")
        report_lines.append("### Готовность групп для анализа:")
        
        for group_name, group_data in analysis_dataset['groups'].items():
            molecules = group_data['molecules']
            target_count = self.optimized_groups[group_name]['target_count']
            
            if len(molecules) >= target_count:
                status = "✅ ГОТОВА"
            elif len(molecules) >= target_count * 0.7:
                status = "⚠️ ЧАСТИЧНО ГОТОВА"
            else:
                status = "❌ НЕ ГОТОВА"
            
            report_lines.append(f"- **{group_name.upper()}**: {status} ({len(molecules)}/{target_count} молекул)")
        
        report_lines.append("")
        report_lines.append("### Следующие шаги:")
        report_lines.append("1. Загрузить обученную EGNN модель")
        report_lines.append("2. Адаптировать модель для предсказания binding affinity")
        report_lines.append("3. Провести анализ domain shift по размерам молекул")
        report_lines.append("4. Оценить точность предсказаний для каждой группы")
        report_lines.append("5. Создать рекомендации по применимости модели")
        
        # Сохраняем отчет
        report_text = "\n".join(report_lines)
        report_file = self.results_dir / "antibacterial_analysis_report.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        logger.info(f"📝 Отчет сохранен: {report_file}")
        return str(report_file)
    
    def run_complete_analysis(self):
        """Запускает полный анализ антибактериальных препаратов."""
        
        logger.info("🚀 Запуск полного анализа антибактериальных препаратов")
        
        try:
            # 1. Подготовка датасета
            logger.info("\n" + "="*60)
            logger.info("📋 ЭТАП 1: ПОДГОТОВКА ДАТАСЕТА")
            logger.info("="*60)
            
            analysis_dataset = self.prepare_analysis_dataset()
            
            # 2. Создание визуализации
            logger.info("\n" + "="*60)
            logger.info("📊 ЭТАП 2: СОЗДАНИЕ ВИЗУАЛИЗАЦИИ")
            logger.info("="*60)
            
            self.create_visualization_summary(analysis_dataset)
            
            # 3. Создание отчета
            logger.info("\n" + "="*60)
            logger.info("📝 ЭТАП 3: СОЗДАНИЕ ОТЧЕТА")
            logger.info("="*60)
            
            report_file = self.create_detailed_report(analysis_dataset)
            
            # 4. Итоговая сводка
            logger.info("\n" + "="*60)
            logger.info("✅ АНАЛИЗ ЗАВЕРШЕН")
            logger.info("="*60)
            
            logger.info(f"📊 Проанализировано: {analysis_dataset['metadata']['total_molecules']} молекул")
            logger.info(f"📁 Результаты сохранены в: {self.results_dir}")
            logger.info(f"📝 Отчет: {report_file}")
            logger.info(f"📊 Визуализация: {self.results_dir / 'antibacterial_summary_visualization.png'}")
            logger.info(f"📋 Датасет: {self.results_dir / 'prepared_antibacterial_dataset.json'}")
            
            # Выводим краткую сводку по группам
            logger.info(f"\n📈 КРАТКАЯ СВОДКА ПО ГРУППАМ:")
            
            for group_name, group_data in analysis_dataset['groups'].items():
                molecules = group_data['molecules']
                target_count = self.optimized_groups[group_name]['target_count']
                
                if len(molecules) >= target_count:
                    status = "✅"
                elif len(molecules) >= target_count * 0.7:
                    status = "⚠️"
                else:
                    status = "❌"
                
                logger.info(f"  {status} {group_name.upper()}: {len(molecules)}/{target_count} молекул")
            
            logger.info(f"\n🎯 Система готова для ML анализа!")
            
            return analysis_dataset
            
        except Exception as e:
            logger.error(f"❌ Ошибка в анализе: {e}")
            raise


def main():
    """Главная функция."""
    
    try:
        # Создаем анализатор
        analyzer = OptimizedAntibacterialAnalysis()
        
        # Запускаем полный анализ
        analysis_dataset = analyzer.run_complete_analysis()
        
        return analysis_dataset
        
    except Exception as e:
        logger.error(f"❌ Ошибка в main: {e}")
        raise


if __name__ == "__main__":
    main()