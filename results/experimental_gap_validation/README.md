# Экспериментальная валидация EGNN модели на антибактериальных препаратах

## Описание

Результаты валидации Equivariant Graph Neural Network (EGNN) модели на реальных экспериментальных данных HOMO-LUMO Gap для 65 антибактериальных препаратов.

## Структура файлов

### raw_data/
- `antibacterial_molecules_dataset.json` - Датасет из 65 антибактериальных молекул с экспериментальными HOMO-LUMO Gap значениями

### validation_results/
- `experimental_validation_predictions.json` - Предсказания модели для всех молекул
- `experimental_validation_metrics.json` - Детальные метрики производительности
- `experimental_validation_results.csv` - Табличные результаты для анализа

### visualizations/
- `correlation_analysis.png` - Корреляция между предсказанными и экспериментальными значениями
- `domain_shift_analysis.png` - Анализ domain shift по размерам молекул
- `comprehensive_summary.png` - Комплексный отчет с ключевыми метриками

### reports/
- `experimental_validation_report.md` - Полный научный отчет с результатами и выводами
- `executive_summary.md` - Краткое резюме ключевых результатов и рекомендаций

## Ключевые результаты

- **MAE**: 0.373 eV (Domain Shift Factor: 4.9x)
- **R²**: 0.505 (статистически значимая корреляция)
- **Pearson r**: 0.711 (p < 1e-10)
- **Молекул**: 65 антибактериальных препаратов

## Методология

Валидация проводилась с использованием реальной EGNN Model 3, обученной на QM9 датасете. Экспериментальные данные собраны из научной литературы с проверкой качества источников.

