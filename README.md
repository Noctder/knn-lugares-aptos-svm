# ¿Puede un robot aprender dónde trabajar? KNN desde cero vs SVM

Proyecto de la primera evaluación parcial de **Aprendizaje Artificial 2026**.

| | |
|---|---|
| **Integrantes (Equipo 4)** | Francisco Javier Zuñiga Milanez, Hugo Rafael Preciado Marquez |
| **Profesor** | Pedro Cesar Santana Mancilla |
| **Materia** | Aprendizaje Artificial |
| **Maestría** | Aprendizaje Artificial, Universidad de Colima |
| **Modelo asignado** | Support Vector Machine (SVM) |

## Descripción del problema

En la primera actividad del curso se definieron reglas manuales para decidir si un lugar del campus es **Apto** o **No apto** para que un robot trabaje durante 30 minutos. En este proyecto se retoma el mismo problema con Machine Learning: se implementa un clasificador **K-Nearest Neighbors (KNN) desde cero**, se compara con el KNN de scikit-learn y con un **SVM**, y se analiza qué decisiones sobre los datos y el modelo afectan los resultados.

## Dataset

Se usa el **Dataset v1.0** construido por todo el grupo y publicado en Kaggle:
[Lugares aptos para trabajar](https://www.kaggle.com/datasets/schiaffino89/lugares-aptos-para-trabajar) (archivo `Dataset v1.0.csv`). En la carpeta `data/` se incluye una copia de esa misma versión.

- 65 lugares y 11 variables binarias (1 = tiene la característica, 0 = no la tiene): `silencio`, `sombra`, `aire_acondicionado`, `soporte_para_trabajar`, `asientos`, `asientos_comodos`, `enchufes`, `flujo_personas`, `banos_cerca`, `cafeteria_cerca`, `internet_wifi`.
- Variable objetivo `Target`: 1 = Apto (34 lugares, 52.3%), 0 = No apto (31 lugares, 47.7%).
- La columna `Lugar` solo es el nombre del sitio y no se usa para clasificar.

## Estructura del repositorio

```
├── README.md
├── requirements.txt
├── Proyecto_InteligenciaArtificial.ipynb   # notebook con todo el proyecto
└── data/
    └── Dataset v1.0.csv         # copia del Dataset v1.0
```

## Cómo ejecutar el proyecto

**Opción 1: Google Colab (recomendada)**

1. Abrir el notebook en Colab (botón *Open in Colab* de GitHub o subir el archivo `.ipynb`).
2. Ejecutar todas las celdas en orden (*Entorno de ejecución > Ejecutar todas*).
3. El dataset se descarga solo desde Kaggle con `kagglehub`.

**Opción 2: Local**

```bash
pip install -r requirements.txt
jupyter notebook Proyecto_InteligenciaArtificial.ipynb
```

Si no hay conexión a Kaggle, en la celda 1 se puede cargar la copia local:

```python
df = pd.read_csv('data/Dataset v1.0.csv')
```

## Metodología

1. **Análisis inicial:** revisión de nulos, duplicados, balance de clases y correlación de cada variable con `Target`.
2. **KNN desde cero:** distancia euclidiana, ordenar distancias, elegir los k vecinos y votar por mayoría. No se usa `KNeighborsClassifier`.
3. **División:** 80% entrenamiento (52 lugares) y 20% prueba (13 lugares), con `random_state=42` y `stratify`. La misma división se usa en todos los experimentos.
4. **Experimentos:** KNN propio con k = 1, 3, 5, 7 y 9; comparación con `KNeighborsClassifier`; y SVM con kernel lineal.
5. **Métrica:** exactitud (accuracy) en el conjunto de prueba para los tres modelos.

## Principales resultados

**KNN propio según k**

| k | Exactitud entrenamiento | Exactitud prueba |
|---|---|---|
| 1 | 1.000 | 0.923 |
| 3 | 0.923 | 0.923 |
| 5 | 0.962 | 0.923 |
| 7 | 0.942 | 0.923 |
| 9 | 0.923 | 0.923 |

**Comparación final (k = 5)**

| Modelo | Exactitud en prueba | Lugares mal clasificados |
|---|---|---|
| KNN propio | 0.923 | Mesas al lado del starvocho |
| KNN scikit-learn | 0.923 | Mesas al lado del starvocho |
| SVM (lineal) | 1.000 | Ninguno |

- El KNN propio y el de scikit-learn dieron exactamente las mismas predicciones con todos los valores de k.
- `soporte_para_trabajar` es la variable más importante (correlación de 0.91 con `Target`): ningún lugar sin soporte es Apto.
- El SVM le dio peso casi solo a `soporte_para_trabajar` y `sombra`, mientras que KNN trata las 11 variables igual. Eso explica por qué el SVM acierta el lugar en el que KNN falla.
- Con solo 13 lugares de prueba, cada error mueve la exactitud unos 7.7 puntos, así que la diferencia entre KNN y SVM (un lugar) debe tomarse con cautela.
