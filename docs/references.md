# Литература и ресурсы

## 📚 Основная литература

### Геометрическое машинное обучение

**Книги и обзоры:**
1. Bronstein, M. M., et al. (2021). *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges*. arXiv:2104.13478
   - 🌟 **Основополагающая работа** по геометрическому DL
   - 📖 [Веб-версия](https://geometricdeeplearning.com/)

2. Cohen, T., & Welling, M. (2016). *Group equivariant convolutional networks*. ICML 2016
   - 🎯 Введение в эквивариантные сверточные сети

3. Weiler, M., & Cesa, G. (2019). *General E(2)-equivariant steerable CNNs*. NeurIPS 2019
   - 🔧 Практическая реализация эквивариантных сетей

### E(n) Equivariant Graph Neural Networks

**Ключевые статьи:**
1. Satorras, V. G., et al. (2021). *E(n) Equivariant Graph Neural Networks*. ICML 2021
   - 📄 [arXiv:2102.09844](https://arxiv.org/abs/2102.09844)
   - 🌟 **Основная статья по EGNN**

2. Schütt, K., et al. (2021). *Equivariant message passing for the prediction of tensorial properties and molecular spectra*. ICML 2021
   - 🧪 Применение к молекулярным свойствам

### Топологический анализ данных

**Основы TDA:**
1. Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction*
   - 📖 Классический учебник по вычислительной топологии

2. Carlsson, G. (2009). *Topology and data*. Bulletin of the American Mathematical Society
   - 🎯 Введение в TDA для анализа данных

**TDA в молекулярной биологии:**
3. Cang, Z., & Wei, G. W. (2017). *TopologyNet: Topology based deep convolutional and multi-task neural networks for biomolecular property predictions*. PLoS computational biology
   - 🧬 Применение топологии к биомолекулам

4. Meng, Z., et al. (2021). *Persistent spectral–based machine learning (PerSpect ML) for protein-ligand binding affinity prediction*. Science advances
   - 💊 TDA для предсказания аффинности связывания

## 🎥 Видеолекции и курсы

### Геометрическое ML
1. **Tess Smidt - Euclidean Neural Networks**
   - 🎥 [YouTube](https://www.youtube.com/watch?v=PtA0lg_e5nA)
   - 🌟 Отличное введение в E(n) эквивариантность

2. **Maurice Weiler - Equivariant Networks**
   - 📖 [Blog post](https://maurice-weiler.gitlab.io/blog_post/cnn-book_1_equivariant_networks/)
   - 🎯 Подробное объяснение с примерами

3. **Michael Bronstein - Geometric Deep Learning**
   - 🎥 [Курс лекций](https://www.youtube.com/playlist?list=PLn2-dEmQeTfQ8YVuHBOvAhUlnIPYxkeu3)
   - 📚 Полный курс по геометрическому DL

### Топологический анализ данных
4. **Gunnar Carlsson - TDA**
   - 🎥 [Stanford lectures](https://www.youtube.com/watch?v=h0bnG1Wavag)
   - 🎓 Академический курс по TDA

## 🧪 Датасеты

### Молекулярные датасеты
1. **QM9**
   - 📊 134k малых органических молекул
   - 🔗 [MoleculeNet](http://moleculenet.ai/datasets-1)
   - 🎯 Квантово-химические свойства

2. **MD17**
   - 📊 Молекулярная динамика для 8 молекул
   - 🔗 [SGDml](http://quantum-machine.org/gdml/)
   - 🎯 Энергии и силы

### Белок-лигандные комплексы
3. **PDBbind**
   - 📊 >19k белок-лигандных комплексов
   - 🔗 [Official site](http://www.pdbbind.org.cn/)
   - 🎯 Экспериментальные аффинности связывания

4. **SAbDab (Structural Antibody Database)**
   - 📊 Структуры антител
   - 🔗 [SAbDab](http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/)
   - 🎯 Антиген-антитело комплексы

## 🛠️ Инструменты и библиотеки

### Геометрическое ML
1. **PyTorch Geometric**
   - 🔗 [GitHub](https://github.com/pyg-team/pytorch_geometric)
   - 🎯 Графовые нейронные сети

2. **e3nn**
   - 🔗 [GitHub](https://github.com/e3nn/e3nn)
   - 🎯 E(3) эквивариантные сети

3. **EGNN PyTorch**
   - 🔗 [GitHub](https://github.com/lucidrains/egnn-pytorch)
   - 🎯 Простая реализация EGNN

### Топологический анализ
4. **GUDHI**
   - 🔗 [Official site](https://gudhi.inria.fr/)
   - 🎯 Библиотека для TDA

5. **Ripser**
   - 🔗 [GitHub](https://github.com/Ripser/ripser)
   - 🎯 Быстрое вычисление персистентной гомологии

6. **Persim**
   - 🔗 [GitHub](https://github.com/scikit-tda/persim)
   - 🎯 Анализ диаграмм персистентности

### Молекулярная химия
7. **RDKit**
   - 🔗 [Official site](https://www.rdkit.org/)
   - 🎯 Хемоинформатика

8. **MDAnalysis**
   - 🔗 [Official site](https://www.mdanalysis.org/)
   - 🎯 Анализ молекулярной динамики

9. **BioPython**
   - 🔗 [Official site](https://biopython.org/)
   - 🎯 Биоинформатика

## 📄 Статьи по drug design

### Геометрическое ML в drug design
1. Stärk, H., et al. (2022). *EquiBind: Geometric Deep Learning for Drug Binding Structure Prediction*. ICML 2022
   - 🎯 Предсказание поз связывания

2. Corso, G., et al. (2022). *DiffDock: Diffusion Steps, Twists, and Turns for Molecular Docking*. arXiv:2210.01776
   - 🔄 Диффузионные модели для докинга

### Антитела и белок-белковые взаимодействия
3. Ruffolo, J. A., et al. (2021). *Geometric potentials from deep learning improve prediction of CDR H3 loop structures*. Bioinformatics
   - 🧬 Геометрическое ML для антител

4. Shan, S., et al. (2022). *Deep learning guided optimization of human antibody against SARS-CoV-2 variants with broad neutralization*. PNAS
   - 🦠 Оптимизация антител с помощью DL

## 🌐 Онлайн ресурсы

### Блоги и туториалы
1. **Distill.pub - Geometric Deep Learning**
   - 🔗 [Distill](https://distill.pub/)
   - 🎨 Интерактивные объяснения

2. **Towards Data Science - TDA**
   - 🔗 [Medium](https://towardsdatascience.com/tagged/topological-data-analysis)
   - 📝 Практические туториалы

### Конференции и воркшопы
3. **ICML Workshop on Computational Biology**
   - 🎪 Ежегодный воркшоп

4. **NeurIPS Workshop on Machine Learning for Structural Biology**
   - 🧬 Специализированный воркшоп

## 📊 Бенчмарки и соревнования

1. **Open Graph Benchmark (OGB)**
   - 🔗 [Official site](https://ogb.stanford.edu/)
   - 🏆 Стандартные бенчмарки для графов

2. **MoleculeNet**
   - 🔗 [Official site](http://moleculenet.ai/)
   - 🧪 Бенчмарки для молекулярного ML

3. **CASP (Critical Assessment of Structure Prediction)**
   - 🔗 [Official site](https://predictioncenter.org/)
   - 🏆 Соревнование по предсказанию структуры белков

## 📚 Дополнительная литература

### Математические основы
1. **Группы и симметрии**
   - Tinkham, M. *Group Theory and Quantum Mechanics*
   - 🎯 Теория групп в физике

2. **Топология**
   - Munkres, J. R. *Topology*
   - 📖 Классический учебник по топологии

### Машинное обучение
3. **Графовые нейронные сети**
   - Wu, Z., et al. (2020). *A comprehensive survey on graph neural networks*. IEEE TNNLS
   - 📊 Обзор GNN методов

---

## 🔄 Обновления

Этот список литературы будет обновляться по мере прохождения проекта и появления новых релевантных работ.

*Последнее обновление: Шаг 1 - Теоретические основы*