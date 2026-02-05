# Failure Analysis Report
## Overview
Анализ ошибок и неудачных предсказаний для всех моделей.

### FCNN
- **Test MAE**: 0.7562 eV
- **Test RMSE**: 0.9297 eV
- **Test R²**: 0.4811
- **Max Error**: 4.8895 eV
- **Mean Error**: 0.0514 eV

**Failure Analysis:**
- Failure threshold (20% of mean): 1.3708 eV
- Estimated failure rate: 11.0%

### EGNN
- **Test MAE**: 0.1107 eV
- **Test RMSE**: 0.1469 eV
- **Test R²**: 0.9870
- **Max Error**: 1.2269 eV
- **Mean Error**: 0.0072 eV

**Failure Analysis:**
- Failure threshold (20% of mean): 1.3708 eV
- Estimated failure rate: 1.6%

### TABULAR
- **Test MAE**: 0.5890 eV
- **Test RMSE**: 0.7629 eV
- **Test R²**: 0.6506
- **Max Error**: 5.1695 eV
- **Mean Error**: 0.0055 eV

**Failure Analysis:**
- Failure threshold (20% of mean): 1.3708 eV
- Estimated failure rate: 8.6%

### GCN
- **Test MAE**: 0.4746 eV
- **Test RMSE**: 0.6262 eV
- **Test R²**: 0.7646
- **Max Error**: 4.5000 eV
- **Mean Error**: 0.0100 eV

**Failure Analysis:**
- Failure threshold (20% of mean): 1.3708 eV
- Estimated failure rate: 6.9%

