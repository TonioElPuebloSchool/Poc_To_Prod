import unittest
from unittest.mock import MagicMock, patch
import os
from run import TextPredictionModel

class TestTextPredictionModel(unittest.TestCase):
    def setUp(self):
        # Set up the test environment, e.g., create a temporary directory for testing
        self.test_artefacts_path = "/path/to/test/artefacts"
        os.makedirs(self.test_artefacts_path, exist_ok=True)

    def tearDown(self):
        # Clean up the test environment, e.g., remove temporary directories
        if os.path.exists(self.test_artefacts_path):
            os.rmdir(self.test_artefacts_path)

    def test_from_artefacts_loads_model_params_and_labels(self):
        # Mocking load_model, open, and json.load to avoid actual file reading and model loading
        with patch("run.load_model", return_value=MagicMock()) as mock_load_model, \
             patch("run.open", create=True) as mock_open, \
             patch("json.load", return_value={"param_key": "param_value"}) as mock_json_load:

            # Create a dummy artefacts directory with required files
            with open(os.path.join(self.test_artefacts_path, "model.h5"), "w") as f:
                f.write("dummy_model_data")

            with open(os.path.join(self.test_artefacts_path, "params.json"), "w") as f:
                f.write('{"param_key": "param_value"}')

            with open(os.path.join(self.test_artefacts_path, "labels_index.json"), "w") as f:
                f.write('{"label1": 0, "label2": 1}')

            # Call the method under test
            model = TextPredictionModel.from_artefacts(self.test_artefacts_path)

            # Assertions
            mock_load_model.assert_called_once_with(os.path.join(self.test_artefacts_path, 'model.h5'))
            mock_json_load.assert_any_call(mock_open(os.path.join(self.test_artefacts_path, 'params.json'), 'r'))
            mock_json_load.assert_any_call(mock_open(os.path.join(self.test_artefacts_path, 'labels_index.json'), 'r'))

            self.assertIsInstance(model, TextPredictionModel)
            self.assertEqual(model.params, {"param_key": "param_value"})
            self.assertEqual(model.labels_to_index, {"label1": 0, "label2": 1})

    def test_predict_calls_embedding_and_model_predict(self):
        # Mocking embed and model.predict
        with patch("run.embed") as mock_embed, \
             patch.object(TextPredictionModel, 'model', spec=True) as mock_model:

            # Set up a test instance of TextPredictionModel
            model = TextPredictionModel(mock_model, {"param_key": "param_value"}, {"label1": 0, "label2": 1})

            # Call the method under test
            model.predict(["test_text"])

            # Assertions
            mock_embed.assert_called_once_with(["test_text"])
            mock_model.predict.assert_called_once()

    def test_predict_with_mock_model(self):
        # Mocking embed and model.predict with a mock model
        with patch("run.embed") as mock_embed, \
             patch.object(TextPredictionModel, 'model', spec=True) as mock_model:

            # Set up a test instance of TextPredictionModel with a mock model
            model = TextPredictionModel(mock_model, {"param_key": "param_value"}, {"label1": 0, "label2": 1})

            # Call the method under test
            predictions = model.predict(["mock_text"], top_k=2)

            # Assertions
            mock_embed.assert_called_once_with(["mock_text"])
            mock_model.predict.assert_called_once()

            # Ensure the predictions are in the expected format
            self.assertIsInstance(predictions, list)
            self.assertEqual(len(predictions), 1)
            self.assertIsInstance(predictions[0], list)
            self.assertEqual(len(predictions[0]), 2)

if __name__ == "__main__":
    unittest.main()
