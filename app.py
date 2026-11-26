"""Flask carga una sola vez el SVM ya entrenado."""
from flask import Flask, render_template, request
from prediction import FEATURES, load_model, predict_place

app = Flask(__name__)
model = load_model()


@app.route('/', methods=['GET', 'POST'])
def index():
    values = {name: '' for name in FEATURES}
    result = error = None
    status = 200
    if request.method == 'POST':
        values = {name: request.form.get(name, '') for name in FEATURES}
        try:
            result = predict_place(values, model)
        except ValueError as exc:
            error = str(exc)
            status = 400
    return render_template('index.html', features=FEATURES, values=values,
                           result=result, error=error), status


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
