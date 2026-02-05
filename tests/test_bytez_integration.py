import unittest
from unittest.mock import MagicMock, patch
from ml.bytez_interface import BytezClient

class TestBytezIntegration(unittest.TestCase):

    @patch('ml.bytez_interface.Bytez')
    def test_client_initialization(self, mock_bytez):
        """Test that the client initializes with an API key."""
        client = BytezClient(api_key="test-key")
        self.assertEqual(client.api_key, "test-key")
        mock_bytez.assert_called_once_with("test-key")

    @patch('ml.bytez_interface.Bytez')
    def test_run_qa_success(self, mock_bytez):
        """Test that run_qa calls the SDK correctly and returns results."""
        # Setup mocks
        mock_sdk_instance = mock_bytez.return_value
        mock_model = MagicMock()
        mock_bytez.return_value.model.return_value = mock_model

        mock_results = MagicMock()
        mock_results.error = None
        mock_results.output = "London"
        mock_model.run.return_value = mock_results

        # Execute
        client = BytezClient(api_key="test-key")
        res = client.run_qa("My name is Simon and I live in London", "Where do I live?")

        # Verify
        mock_sdk_instance.model.assert_called_with("charlieCs/Qwen-14B-dacon-qa")
        mock_model.run.assert_called_once()
        self.assertEqual(res["output"], "London")
        self.assertIsNone(res["error"])

    @patch('ml.bytez_interface.Bytez')
    def test_run_model_error_handling(self, mock_bytez):
        """Test that exceptions are caught and returned as errors."""
        mock_model = MagicMock()
        mock_bytez.return_value.model.return_value = mock_model
        mock_model.run.side_effect = Exception("API Connection Error")

        client = BytezClient(api_key="test-key")
        res = client.run_model("some-model", {"input": "test"})

        self.assertIn("API Connection Error", res["error"])
        self.assertIsNone(res["output"])

if __name__ == "__main__":
    unittest.main()
