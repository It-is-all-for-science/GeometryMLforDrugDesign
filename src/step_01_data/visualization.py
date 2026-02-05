"""
Модуль для визуализации молекулярных данных и статистик.

Содержит функции для создания интерактивных и статических визуализаций
молекулярных структур, распределений свойств и статистического анализа.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from torch_geometric.data import Data
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Настройка стиля matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MolecularVisualizer:
    """
    Класс для визуализации молекулярных структур и данных.
    
    Поддерживает как 2D, так и 3D визуализации с интерактивными элементами.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Args:
            figsize: Размер фигур по умолчанию
        """
        self.figsize = figsize
        
        # Цветовая схема для атомов (по атомным номерам)
        self.atom_colors = {
            1: '#FFFFFF',   # H - белый
            6: '#909090',   # C - серый
            7: '#3050F8',   # N - синий
            8: '#FF0D0D',   # O - красный
            9: '#90E050',   # F - зеленый
            15: '#FF8000',  # P - оранжевый
            16: '#FFFF30',  # S - желтый
            17: '#1FF01F',  # Cl - зеленый
            35: '#A62929',  # Br - коричневый
            53: '#940094',  # I - фиолетовый
        }
        
        # Размеры атомов (ван-дер-ваальсовы радиусы в Å)
        self.atom_sizes = {
            1: 1.2,   # H
            6: 1.7,   # C
            7: 1.55,  # N
            8: 1.52,  # O
            9: 1.47,  # F
            15: 1.8,  # P
            16: 1.8,  # S
            17: 1.75, # Cl
            35: 1.85, # Br
            53: 1.98, # I
        }
    
    def plot_molecule_3d(self, 
                        data: Data, 
                        title: str = "Молекулярная структура",
                        show_bonds: bool = True,
                        show_labels: bool = False) -> go.Figure:
        """
        Создает 3D визуализацию молекулы.
        
        Args:
            data: Молекулярный граф с координатами
            title: Заголовок графика
            show_bonds: Показывать ли связи
            show_labels: Показывать ли подписи атомов
        
        Returns:
            plotly.graph_objects.Figure: 3D график молекулы
        """
        if not hasattr(data, 'pos') or data.pos is None:
            raise ValueError("Данные должны содержать 3D координаты (pos)")
        
        pos = data.pos.numpy()
        
        # Получаем атомные номера
        if hasattr(data, 'z'):
            atomic_numbers = data.z.numpy()
        else:
            atomic_numbers = np.ones(pos.shape[0])  # По умолчанию водород
        
        # Создаем 3D scatter plot для атомов
        atom_traces = []
        
        for atom_num in np.unique(atomic_numbers):
            mask = atomic_numbers == atom_num
            atom_pos = pos[mask]
            
            color = self.atom_colors.get(atom_num, '#808080')
            size = self.atom_sizes.get(atom_num, 1.5) * 10  # Масштабируем для визуализации
            
            # Название элемента
            element_name = self._get_element_name(atom_num)
            
            trace = go.Scatter3d(
                x=atom_pos[:, 0],
                y=atom_pos[:, 1],
                z=atom_pos[:, 2],
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    opacity=0.8,
                    line=dict(width=2, color='black')
                ),
                name=f'{element_name} ({int(atom_num)})',
                text=[f'{element_name}{i}' for i in range(len(atom_pos))],
                hovertemplate='<b>%{text}</b><br>' +
                             'X: %{x:.3f}<br>' +
                             'Y: %{y:.3f}<br>' +
                             'Z: %{z:.3f}<extra></extra>'
            )
            atom_traces.append(trace)
        
        # Создаем фигуру
        fig = go.Figure(data=atom_traces)
        
        # Добавляем связи, если требуется
        if show_bonds and hasattr(data, 'edge_index'):
            edge_index = data.edge_index.numpy()
            
            bond_x, bond_y, bond_z = [], [], []
            
            for i in range(edge_index.shape[1]):
                start_idx, end_idx = edge_index[:, i]
                
                # Координаты начала и конца связи
                start_pos = pos[start_idx]
                end_pos = pos[end_idx]
                
                bond_x.extend([start_pos[0], end_pos[0], None])
                bond_y.extend([start_pos[1], end_pos[1], None])
                bond_z.extend([start_pos[2], end_pos[2], None])
            
            # Добавляем связи как линии
            bond_trace = go.Scatter3d(
                x=bond_x,
                y=bond_y,
                z=bond_z,
                mode='lines',
                line=dict(color='gray', width=3),
                name='Связи',
                showlegend=False,
                hoverinfo='skip'
            )
            fig.add_trace(bond_trace)
        
        # Настройка макета
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (Å)',
                yaxis_title='Y (Å)',
                zaxis_title='Z (Å)',
                aspectmode='cube',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_property_distribution(self, 
                                 target_values: torch.Tensor,
                                 property_name: str,
                                 units: str = "",
                                 bins: int = 50) -> plt.Figure:
        """
        Визуализирует распределение молекулярного свойства.
        
        Args:
            target_values: Значения свойства
            property_name: Название свойства
            units: Единицы измерения
            bins: Количество бинов для гистограммы
        
        Returns:
            matplotlib.figure.Figure: График распределения
        """
        values = target_values.numpy()
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(f'Распределение {property_name}', fontsize=16)
        
        # Гистограмма
        axes[0, 0].hist(values, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_xlabel(f'{property_name} ({units})')
        axes[0, 0].set_ylabel('Частота')
        axes[0, 0].set_title('Гистограмма')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(values, vert=True)
        axes[0, 1].set_ylabel(f'{property_name} ({units})')
        axes[0, 1].set_title('Box Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot (проверка нормальности)
        from scipy import stats
        stats.probplot(values, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot (нормальность)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Кумулятивное распределение
        sorted_values = np.sort(values)
        cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        axes[1, 1].plot(sorted_values, cumulative, linewidth=2)
        axes[1, 1].set_xlabel(f'{property_name} ({units})')
        axes[1, 1].set_ylabel('Кумулятивная вероятность')
        axes[1, 1].set_title('Кумулятивное распределение')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Добавляем статистики как текст
        stats_text = f"""Статистики:
        Среднее: {np.mean(values):.4f}
        Медиана: {np.median(values):.4f}
        Ст. откл.: {np.std(values):.4f}
        Мин: {np.min(values):.4f}
        Макс: {np.max(values):.4f}
        Асимметрия: {stats.skew(values):.4f}
        Эксцесс: {stats.kurtosis(values):.4f}"""
        
        fig.text(0.02, 0.02, stats_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        return fig
    
    def plot_dataset_statistics(self, 
                              data_list: List[Data],
                              target_values: torch.Tensor,
                              metadata: Dict[str, any]) -> plt.Figure:
        """
        Создает обзорную визуализацию статистик датасета.
        
        Args:
            data_list: Список молекулярных графов
            target_values: Целевые значения
            metadata: Метаданные датасета
        
        Returns:
            matplotlib.figure.Figure: Обзорный график
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Статистики датасета {metadata.get("dataset_name", "Unknown")}', 
                    fontsize=16)
        
        # 1. Распределение количества атомов
        num_atoms = [data.x.size(0) if hasattr(data, 'x') else 0 for data in data_list]
        axes[0, 0].hist(num_atoms, bins=30, alpha=0.7, color='lightblue')
        axes[0, 0].set_xlabel('Количество атомов')
        axes[0, 0].set_ylabel('Частота')
        axes[0, 0].set_title('Распределение размеров молекул')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Распределение количества связей
        num_bonds = [data.edge_index.size(1) if hasattr(data, 'edge_index') else 0 
                    for data in data_list]
        axes[0, 1].hist(num_bonds, bins=30, alpha=0.7, color='lightgreen')
        axes[0, 1].set_xlabel('Количество связей')
        axes[0, 1].set_ylabel('Частота')
        axes[0, 1].set_title('Распределение связности')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Соотношение атомы/связи
        atom_bond_ratio = [b/a if a > 0 else 0 for a, b in zip(num_atoms, num_bonds)]
        axes[0, 2].scatter(num_atoms, num_bonds, alpha=0.5, s=20)
        axes[0, 2].set_xlabel('Количество атомов')
        axes[0, 2].set_ylabel('Количество связей')
        axes[0, 2].set_title('Соотношение атомы-связи')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Распределение целевого свойства
        values = target_values.numpy()
        axes[1, 0].hist(values, bins=40, alpha=0.7, color='salmon')
        axes[1, 0].set_xlabel(f'{metadata.get("target_property", "Target")} '
                             f'({metadata.get("property_units", "")})')
        axes[1, 0].set_ylabel('Частота')
        axes[1, 0].set_title('Распределение целевого свойства')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Корреляция размер-свойство
        axes[1, 1].scatter(num_atoms, values, alpha=0.5, s=20)
        axes[1, 1].set_xlabel('Количество атомов')
        axes[1, 1].set_ylabel(f'{metadata.get("target_property", "Target")}')
        axes[1, 1].set_title('Размер молекулы vs Свойство')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Вычисляем корреляцию
        correlation = np.corrcoef(num_atoms, values)[0, 1]
        axes[1, 1].text(0.05, 0.95, f'Корреляция: {correlation:.3f}', 
                       transform=axes[1, 1].transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white"))
        
        # 6. Распределение атомных типов
        if hasattr(data_list[0], 'z'):
            all_atomic_numbers = []
            for data in data_list:
                if hasattr(data, 'z'):
                    all_atomic_numbers.extend(data.z.tolist())
            
            unique_atoms, counts = np.unique(all_atomic_numbers, return_counts=True)
            element_names = [self._get_element_name(z) for z in unique_atoms]
            
            axes[1, 2].bar(element_names, counts, alpha=0.7, color='gold')
            axes[1, 2].set_xlabel('Элемент')
            axes[1, 2].set_ylabel('Количество атомов')
            axes[1, 2].set_title('Распределение элементов')
            axes[1, 2].tick_params(axis='x', rotation=45)
            axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, 'Атомные номера\nнедоступны', 
                           ha='center', va='center', transform=axes[1, 2].transAxes)
        
        plt.tight_layout()
        return fig
    
    def plot_molecular_properties_correlation(self, 
                                            properties: Dict[str, torch.Tensor],
                                            target_values: torch.Tensor,
                                            target_name: str) -> plt.Figure:
        """
        Визуализирует корреляции между молекулярными свойствами.
        
        Args:
            properties: Словарь молекулярных свойств
            target_values: Целевые значения
            target_name: Название целевого свойства
        
        Returns:
            matplotlib.figure.Figure: Матрица корреляций
        """
        # Подготавливаем данные для корреляционного анализа
        data_dict = {}
        
        for prop_name, prop_values in properties.items():
            if prop_values.dim() == 1:  # Только скалярные свойства
                data_dict[prop_name] = prop_values.numpy()
        
        data_dict[target_name] = target_values.numpy()
        
        # Создаем DataFrame
        df = pd.DataFrame(data_dict)
        
        # Вычисляем корреляционную матрицу
        correlation_matrix = df.corr()
        
        # Создаем тепловую карту
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True,
                   fmt='.3f',
                   cbar_kws={"shrink": .8},
                   ax=ax)
        
        ax.set_title('Корреляции между молекулярными свойствами', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def create_interactive_dashboard(self, 
                                   data_list: List[Data],
                                   target_values: torch.Tensor,
                                   metadata: Dict[str, any]) -> go.Figure:
        """
        Создает интерактивный дашборд для исследования данных.
        
        Args:
            data_list: Список молекулярных графов
            target_values: Целевые значения
            metadata: Метаданные датасета
        
        Returns:
            plotly.graph_objects.Figure: Интерактивный дашборд
        """
        # Подготавливаем данные
        num_atoms = [data.x.size(0) if hasattr(data, 'x') else 0 for data in data_list]
        num_bonds = [data.edge_index.size(1) if hasattr(data, 'edge_index') else 0 
                    for data in data_list]
        values = target_values.numpy()
        
        # Создаем subplot с несколькими панелями
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Распределение целевого свойства', 
                          'Размер молекул vs Свойство',
                          'Распределение размеров молекул',
                          'Связность молекул'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Гистограмма целевого свойства
        fig.add_trace(
            go.Histogram(x=values, nbinsx=40, name='Распределение', 
                        marker_color='lightblue', opacity=0.7),
            row=1, col=1
        )
        
        # 2. Scatter plot размер vs свойство
        fig.add_trace(
            go.Scatter(x=num_atoms, y=values, mode='markers',
                      name='Молекулы', marker=dict(size=5, opacity=0.6),
                      hovertemplate='Атомов: %{x}<br>Свойство: %{y:.4f}<extra></extra>'),
            row=1, col=2
        )
        
        # 3. Распределение размеров молекул
        fig.add_trace(
            go.Histogram(x=num_atoms, nbinsx=30, name='Размеры',
                        marker_color='lightgreen', opacity=0.7),
            row=2, col=1
        )
        
        # 4. Scatter plot атомы vs связи
        fig.add_trace(
            go.Scatter(x=num_atoms, y=num_bonds, mode='markers',
                      name='Связность', marker=dict(size=5, opacity=0.6),
                      hovertemplate='Атомов: %{x}<br>Связей: %{y}<extra></extra>'),
            row=2, col=2
        )
        
        # Обновляем макет
        fig.update_layout(
            title_text=f"Интерактивный анализ датасета {metadata.get('dataset_name', 'Unknown')}",
            showlegend=False,
            height=800
        )
        
        # Подписи осей
        fig.update_xaxes(title_text=f"{metadata.get('target_property', 'Target')} "
                                  f"({metadata.get('property_units', '')})", 
                        row=1, col=1)
        fig.update_xaxes(title_text="Количество атомов", row=1, col=2)
        fig.update_xaxes(title_text="Количество атомов", row=2, col=1)
        fig.update_xaxes(title_text="Количество атомов", row=2, col=2)
        
        fig.update_yaxes(title_text="Частота", row=1, col=1)
        fig.update_yaxes(title_text=f"{metadata.get('target_property', 'Target')}", row=1, col=2)
        fig.update_yaxes(title_text="Частота", row=2, col=1)
        fig.update_yaxes(title_text="Количество связей", row=2, col=2)
        
        return fig
    
    def save_visualizations(self, 
                          figures: List[Union[plt.Figure, go.Figure]], 
                          filenames: List[str],
                          output_dir: str = "results/figures") -> None:
        """
        Сохраняет визуализации в файлы.
        
        Args:
            figures: Список фигур для сохранения
            filenames: Список имен файлов
            output_dir: Директория для сохранения
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for fig, filename in zip(figures, filenames):
            filepath = output_path / filename
            
            if isinstance(fig, plt.Figure):
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"Matplotlib фигура сохранена: {filepath}")
            elif isinstance(fig, go.Figure):
                # Сохраняем как HTML для интерактивности
                html_path = filepath.with_suffix('.html')
                fig.write_html(str(html_path))
                
                # Также сохраняем как PNG
                png_path = filepath.with_suffix('.png')
                fig.write_image(str(png_path), width=1200, height=800)
                
                logger.info(f"Plotly фигура сохранена: {html_path} и {png_path}")
    
    def _get_element_name(self, atomic_number: int) -> str:
        """Возвращает символ элемента по атомному номеру."""
        element_symbols = {
            1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O',
            9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
            16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 35: 'Br', 53: 'I'
        }
        return element_symbols.get(int(atomic_number), f'X{int(atomic_number)}')


def create_comprehensive_visualization_report(data_list: List[Data],
                                            target_values: torch.Tensor,
                                            metadata: Dict[str, any],
                                            output_dir: str = "results/figures") -> Dict[str, str]:
    """
    Создает полный отчет с визуализациями для молекулярного датасета.
    
    Args:
        data_list: Список молекулярных графов
        target_values: Целевые значения
        metadata: Метаданные датасета
        output_dir: Директория для сохранения
    
    Returns:
        Dict[str, str]: Словарь с путями к созданным файлам
    """
    visualizer = MolecularVisualizer()
    
    logger.info("Создание комплексного отчета с визуализациями...")
    
    # Создаем различные визуализации
    figures = []
    filenames = []
    
    # 1. Обзорная статистика датасета
    dataset_stats_fig = visualizer.plot_dataset_statistics(data_list, target_values, metadata)
    figures.append(dataset_stats_fig)
    filenames.append(f"{metadata.get('dataset_name', 'dataset')}_overview.png")
    
    # 2. Распределение целевого свойства
    property_dist_fig = visualizer.plot_property_distribution(
        target_values, 
        metadata.get('target_property', 'Target'),
        metadata.get('property_units', '')
    )
    figures.append(property_dist_fig)
    filenames.append(f"{metadata.get('target_property', 'target')}_distribution.png")
    
    # 3. Интерактивный дашборд
    dashboard_fig = visualizer.create_interactive_dashboard(data_list, target_values, metadata)
    figures.append(dashboard_fig)
    filenames.append(f"{metadata.get('dataset_name', 'dataset')}_dashboard")
    
    # 4. 3D визуализация нескольких молекул (если есть координаты)
    if hasattr(data_list[0], 'pos') and data_list[0].pos is not None:
        # Выбираем несколько интересных молекул для визуализации
        indices_to_visualize = [0, len(data_list)//4, len(data_list)//2, -1]
        
        for i, idx in enumerate(indices_to_visualize):
            if idx < len(data_list):
                mol_fig = visualizer.plot_molecule_3d(
                    data_list[idx], 
                    title=f"Молекула {idx} ({metadata.get('dataset_name', 'Unknown')})"
                )
                figures.append(mol_fig)
                filenames.append(f"molecule_{idx}_3d")
    
    # Сохраняем все визуализации
    visualizer.save_visualizations(figures, filenames, output_dir)
    
    # Создаем словарь с путями к файлам
    file_paths = {}
    output_path = Path(output_dir)
    
    for filename in filenames:
        if filename.endswith('.png'):
            file_paths[filename.replace('.png', '')] = str(output_path / filename)
        else:
            # Для plotly файлов
            file_paths[f"{filename}_html"] = str(output_path / f"{filename}.html")
            file_paths[f"{filename}_png"] = str(output_path / f"{filename}.png")
    
    logger.info(f"Создано {len(figures)} визуализаций в директории {output_dir}")
    
    return file_paths