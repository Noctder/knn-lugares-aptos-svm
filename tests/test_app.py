"""Pruebas de contrato e integración con el artefacto real."""
import unittest
from unittest.mock import patch
from prediction import FEATURES, input_frame, load_model, predict_place


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = load_model()
        with patch('sklearn.svm.SVC.fit', side_effect=AssertionError('No entrenar')):
            import app
            cls.web = app
        cls.client = cls.web.app.test_client()

    def test_form_has_exact_columns(self):
        page = self.client.get('/')
        self.assertEqual(page.status_code, 200)
        for name in FEATURES:
            self.assertIn(f'name="{name}"'.encode(), page.data)
        self.assertEqual(page.data.count(b'type="number"'), 11)

    def test_three_predictions_match_saved_model(self):
        rows = [[1,1,1,1,1,1,1,1,1,0,1],
                [0,0,0,0,1,0,0,1,0,0,1],
                [1,0,0,1,1,1,0,1,0,1,1]]
        for row in rows:
            with self.subTest(row=row):
                values = dict(zip(FEATURES, row))
                expected = ('NO APTO', 'APTO')[int(self.model.predict(input_frame(values))[0])]
                self.assertEqual(predict_place(values, self.model), expected)
                with patch('sklearn.svm.SVC.fit', side_effect=AssertionError('No entrenar')):
                    response = self.client.post('/', data=values)
                self.assertEqual(response.status_code, 200)
                self.assertIn(f'<h2>{expected}</h2>'.encode(), response.data)

    def test_invalid_values_never_reach_model(self):
        for bad in ['', 'abc', 'NaN', 'inf', '-1', '2', '0.5']:
            values = dict.fromkeys(FEATURES, '1')
            values['silencio'] = bad
            with self.subTest(value=bad), patch.object(self.web.model, 'predict') as predict:
                self.assertEqual(self.client.post('/', data=values).status_code, 400)
                predict.assert_not_called()
        with patch.object(self.web.model, 'predict') as predict:
            self.assertEqual(self.client.post('/', data={}).status_code, 400)
            predict.assert_not_called()

    def test_response_uses_model_prediction(self):
        values = dict.fromkeys(FEATURES, 1)
        for number, label in [(0, 'NO APTO'), (1, 'APTO')]:
            with patch.object(self.web.model, 'predict', return_value=[number]) as predict:
                response = self.client.post('/', data=values)
                self.assertIn(f'<h2>{label}</h2>'.encode(), response.data)
                predict.assert_called_once()
                self.assertEqual(tuple(predict.call_args.args[0].columns), FEATURES)


if __name__ == '__main__':
    unittest.main()
