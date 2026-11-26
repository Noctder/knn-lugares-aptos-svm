"""Preparación fuera de Flask: reproduce la celda 10 y guarda el SVM."""
import hashlib
import json
from pathlib import Path
import sys
import joblib
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prediction import FEATURES


def main():
    dataset = ROOT / 'data' / 'Dataset v1.0.csv'
    df = pd.read_csv(dataset)
    assert tuple(c for c in df.columns if c not in ('Lugar', 'Target')) == FEATURES
    X, y = df[list(FEATURES)], df['Target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    # Igual que el notebook: variables binarias sin escalado ni imputación.
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    destination = ROOT / 'studybot_model.joblib'
    joblib.dump(model, destination)
    restored = joblib.load(destination)
    assert (model.predict(X_test) == restored.predict(X_test)).all()
    report = {
        'source_notebook': 'Proyecto_InteligenciaArtificial (1).ipynb, celdas 7 y 10',
        'note': 'Artefacto reconstruido con el procedimiento del notebook; no existía un joblib previo.',
        'model': 'SVC(kernel="linear")', 'preprocessing': 'Ninguno; columnas binarias en orden original',
        'features': FEATURES, 'sklearn_version': sklearn.__version__,
        'train_rows': len(X_train), 'test_rows': len(X_test),
        'test_accuracy': restored.score(X_test, y_test), 'reload_predictions_equal': True,
        'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
        'model_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
    }
    (ROOT / 'evidence' / 'model_metadata.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
