"""Registra tres pruebas de la función y del formulario Flask con el modelo real."""
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prediction import FEATURES, input_frame, load_model, predict_place

model = load_model()
df = pd.read_csv(ROOT / 'data' / 'Dataset v1.0.csv')
cases = [
    ('Laboratorio 4 de posgrado', 'Las condiciones del laboratorio coinciden con los patrones aprendidos de lugares aptos.'),
    ('Bancas verdes centro de idiomas', 'El modelo rechaza este lugar exterior con pocos servicios y sin soporte para trabajar.'),
    ('Atrás de servicios', 'Caso discutible: tiene soporte, pero no sombra, aire acondicionado ni enchufes. El modelo coincide con la etiqueta APTO, pero su margen es muy cercano a cero. No debe interpretarse como una garantía de comodidad.'),
]
# Primero se comprueba la función de predicción de forma independiente.
records = []
for name, interpretation in cases:
    row = df.loc[df['Lugar'] == name].iloc[0]
    values = {f: int(row[f]) for f in FEATURES}
    records.append({'lugar': name, 'inputs': values,
                    'prediction': predict_place(values, model),
                    'dataset_target': int(row['Target']),
                    'decision_function': float(model.decision_function(input_frame(values))[0]),
                    'interpretation': interpretation})
from app import app
client = app.test_client()
for i, record in enumerate(records, 1):
    response = client.post('/', data=record['inputs'])
    assert response.status_code == 200
    assert f"<h2>{record['prediction']}</h2>" in response.get_data(as_text=True)
    record['http_status'] = response.status_code
    (ROOT / 'evidence' / f'prueba_{i}.html').write_bytes(response.data)
report = {'model_sha256': hashlib.sha256((ROOT / 'studybot_model.joblib').read_bytes()).hexdigest(),
          'note': 'decision_function es un margen firmado del SVM, no una probabilidad. Cero es la frontera.',
          'tests': records}
(ROOT / 'evidence' / 'predictions.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
for record in records:
    print(record['lugar'], record['prediction'], record['decision_function'])
