# Evidencia de pruebas de StudyBot

Fecha: 26 de noviembre de 2026. Python 3.12, SVM lineal guardado, Flask con cliente de pruebas; tres POST HTTP 200. Estas pruebas son funcionales y no una nueva evaluación independiente de generalización.

## Entradas completas

| Variable | Laboratorio 4 | Bancas verdes | Atrás de servicios |
|---|---:|---:|---:|
| silencio | 1 | 0 | 1 |
| sombra | 1 | 0 | 0 |
| aire_acondicionado | 1 | 0 | 0 |
| soporte_para_trabajar | 1 | 0 | 1 |
| asientos | 1 | 1 | 1 |
| asientos_comodos | 1 | 0 | 1 |
| enchufes | 1 | 0 | 0 |
| flujo_personas | 1 | 1 | 1 |
| banos_cerca | 1 | 0 | 0 |
| cafeteria_cerca | 0 | 0 | 1 |
| internet_wifi | 1 | 1 | 1 |

## Resultados e interpretación

| Caso | Resultado | Margen SVM | Interpretación |
|---|---|---:|---|
| Laboratorio 4 de posgrado | APTO | 1.004384 | Las condiciones del laboratorio coinciden con los patrones aprendidos de lugares aptos. |
| Bancas verdes centro de idiomas | NO APTO | -2.003159 | El modelo rechaza este lugar exterior con pocos servicios y sin soporte para trabajar. |
| Atrás de servicios | APTO | 0.001286 | Caso discutible: tiene soporte, pero no sombra, aire acondicionado ni enchufes. El modelo coincide con la etiqueta APTO, pero su margen es muy cercano a cero. No debe interpretarse como una garantía de comodidad. |

El margen de `decision_function()` tiene frontera en cero; no es una probabilidad. El tercer caso se verificó también desde el formulario en el navegador.

## Validación adicional

- Cuatro pruebas automatizadas aprobadas (incluyen las tres combinaciones).
- Recarga joblib: mismas predicciones antes y después de guardar, sobre las 13 filas de prueba.
- Exactitud reproducida del notebook: 13/13; conjunto de entrenamiento de 52 filas.
- Campos vacíos, texto, NaN, infinito, -1, 2 y 0.5: HTTP 400, sin llamar a predict.
- Las pruebas bloquean SVC.fit durante la carga de Flask y las peticiones.
- Se sustituyó predict temporalmente en una prueba: el HTML sigue la salida del modelo para la misma entrada; no hay reglas manuales de clasificación en app.py.
- Prueba de instalación y reinicio en entorno limpio: ver reproducibility.txt.

Los JSON registran las entradas y el hash SHA-256 del modelo. Las respuestas HTML de los tres POST están guardadas como prueba_1.html, prueba_2.html y prueba_3.html.
