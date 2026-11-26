"""Carga e inferencia; este módulo nunca entrena el modelo."""
from pathlib import Path
import math
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
FEATURES = (
    'silencio', 'sombra', 'aire_acondicionado', 'soporte_para_trabajar',
    'asientos', 'asientos_comodos', 'enchufes', 'flujo_personas',
    'banos_cerca', 'cafeteria_cerca', 'internet_wifi',
)
LABELS = ('NO APTO', 'APTO')


def load_model(path=BASE_DIR / 'studybot_model.joblib'):
    # Cargar únicamente el artefacto de confianza incluido en el repositorio.
    model = joblib.load(path)
    if tuple(model.feature_names_in_) != FEATURES:
        raise ValueError('Las columnas del modelo no coinciden con el formulario.')
    if tuple(model.classes_) != (0, 1):
        raise ValueError('Clases inesperadas en el modelo.')
    return model


def input_frame(values):
    """Valida las 11 entradas y conserva nombres, orden y codificación originales."""
    row = {}
    for name in FEATURES:
        try:
            number = float(values[name])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f'{name}: introduce un valor numérico (0 o 1).') from None
        if not math.isfinite(number) or number not in (0, 1):
            raise ValueError(f'{name}: solo se permite 0 o 1.')
        row[name] = int(number)
    return pd.DataFrame([row], columns=FEATURES)


def predict_place(values, model):
    """Recibe un diccionario y devuelve APTO / NO APTO mediante predict()."""
    return LABELS[int(model.predict(input_frame(values))[0])]
