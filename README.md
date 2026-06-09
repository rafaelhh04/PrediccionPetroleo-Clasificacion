# Predicción del Precio del Brent — Clasificación Binaria

> Proyecto de Aprendizaje Automático | Curso 2025-26

Predicción de la **dirección del precio del petróleo Brent al día siguiente** mediante técnicas de clasificación supervisada, combinando indicadores de mercado con eventos geopolíticos globales (2010–2026).

---

## Descripción del Problema

El objetivo es predecir si el precio del Brent **subirá o bajará** el día siguiente:

| Clase | Condición | Significado |
|-------|-----------|-------------|
| `1` | `brent_return(t+1) > 0` | El precio sube |
| `0` | `brent_return(t+1) ≤ 0` | El precio baja o se mantiene |

El periodo cubierto (2010–2026) incluye eventos de alta relevancia como la pandemia de COVID-19 (2020), la guerra de Ucrania (2022) o la crisis del Mar Rojo (2024), aportando gran variabilidad al dataset.

---

##  Estructura del Proyecto

```
PROYECTOML/
│
├── main.py                        # ← Punto de entrada. Reproduce todos los resultados
│
├── data/
│   ├── load_data.py               # Carga y merge de los dos datasets
│   ├── dataset_pretoleo_1.csv     # ~4.000 registros diarios (2010–2026)
│   └── geopolitical_events_timeline_1.csv  # 35 eventos geopolíticos
│
├── models/
│   ├── logistic_regression.py     # Modelo baseline (sklearn)
│   ├── neural_network.py          # MLP implementado con NumPy puro 
│   ├── svm_model.py               # SVM con kernel RBF (sklearn)
│   └── random_forest.py           # Random Forest (sklearn)
│
├── utils/
│   ├── preprocessing.py           # Limpieza, features, label, split temporal
│   └── evaluation.py              # Métricas, matrices de confusión, curvas ROC
│
├── results/
│   ├── final_results.csv          # Tabla comparativa de métricas
│   └── plots/                     # Todas las gráficas generadas
│
├── environment.yml                # Entorno conda reproducible
└── memoria.pdf                    # Memoria del proyecto
```

---

##  Datasets

### Dataset 1 — Oil Prices & Market Indicators
- **Fuente**: [Kaggle — Global Oil Prices and Geopolitical Events](https://www.kaggle.com/datasets/kavyadhyani/global-oil-prices-andgeopolitical-events)
- **Periodo**: Febrero 2010 – Marzo 2026 (datos diarios)
- **Registros**: ~4.000 filas · 23 columnas

| Grupo | Features |
|-------|----------|
| Precios | `brent_price`, `wti_price` |
| Retornos | `brent_return`, `wti_return` |
| Lags | `brent_lag_1/3/7`, `wti_lag_1/3/7` |
| Volatilidad | `brent_volatility_7d/30d`, `wti_volatility_7d/30d` |
| Macro | `dxy_index`, `vix`, `gpr_index`, `brent_wti_spread` |

### Dataset 2 — Geopolitical Events (2010–2026)
- **Registros**: 35 eventos documentados · 4 columnas
- **Columnas**: `date`, `event_type`, `event_description`, `event_severity` (escala 1–10)
- **Categorías**: `war`, `sanctions`, `opec`, `conflict`, `disaster`, `blockade`, `oil_price_war`, `market_crash`

Los dos datasets se integran mediante **left join** sobre `date`. Los días sin evento registrado reciben `event_type = 'none'` y `event_severity = 0`.

---

##  Modelos Implementados

| Modelo | Implementación | Rol |
|--------|---------------|-----|
| Regresión Logística | scikit-learn | Baseline interpretable |
| **Red Neuronal MLP** | **NumPy puro** ⭐ | Obligatoria por requisitos |
| SVM (kernel RBF) | scikit-learn | Frontera de decisión no lineal |
| Random Forest | scikit-learn | Importancia de features |

### Arquitectura de la Red Neuronal (NumPy)
```
Input  →  [64 neuronas + ReLU + Dropout]  →  [32 neuronas + ReLU + Dropout]  →  Sigmoid
```
Implementa desde cero: forward pass, backpropagation, mini-batch gradient descent, dropout e inicialización He.

---

##  Instalación y Ejecución

### 1. Clonar el repositorio
```bash
git clone <url-del-repo>
cd PROYECTOML
```

### 2. Crear el entorno conda
```bash
conda env create -f environment.yml
conda activate proyectoml
```

### 3. Ejecutar el proyecto completo
```bash
python main.py
```

Una sola ejecución reproduce todos los resultados: carga de datos, EDA, preprocesado, entrenamiento de los 4 modelos, búsqueda de hiperparámetros, evaluación final y generación de gráficas.

---

## Metodología

### División temporal de datos
Se respeta estrictamente el orden cronológico para evitar data leakage:

```
2010 ─────────────── 2021 │ 2022 ──── 2023 │ 2024 ──── 2026
        TRAIN (75%)        │  VAL (15%) │   TEST (10%)
```

### Validación cruzada
`TimeSeriesSplit` con 5 folds preservando el orden temporal.

### Búsqueda de hiperparámetros
- `GridSearchCV` para Regresión Logística
- `RandomizedSearchCV` para SVM y Random Forest
- Búsqueda manual en bucle para la Red Neuronal NumPy

### Métricas de evaluación
`Accuracy` · `F1-Score` · `AUC-ROC` · `Matriz de confusión`

---

##  Resultados

Los resultados completos se encuentran en `results/final_results.csv` tras ejecutar `main.py`.

Las gráficas generadas incluyen:

```
results/plots/
├── eda_label_distribution.png       # Distribución de clases
├── eda_price_history.png            # Evolución temporal Brent/WTI
├── eda_correlation_matrix.png       # Heatmap de correlaciones
├── roc_curve_all_models.png         # Curvas ROC comparativas
├── confusion_matrix_[modelo].png    # Matrices de confusión
├── learning_curve_[modelo].png      # Curvas de entrenamiento
└── feature_importance_rf.png        # Importancia de features (RF)
```

---

##  Restricciones Técnicas

- ✅ Red neuronal implementada **exclusivamente con NumPy**
- ✅ Todo el código en archivos `.py` — sin Jupyter Notebooks
- ✅ `main.py` reproduce todos los resultados con una sola ejecución
- ✅ División de datos respetando orden cronológico estricto
- ✅ `random_state=42` en todas las operaciones aleatorias

---

##  Dependencias

```
Python  3.11
numpy
pandas
scikit-learn
matplotlib
seaborn
```

---

##  Autor
Rafael Hernando Herias
Proyecto individual — Asignatura de Aprendizaje Automático | Curso 2025-26
