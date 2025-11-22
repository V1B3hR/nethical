"""Tests for manifest file validation."""

import json
import yaml
import pytest
from pathlib import Path


# Base directory for the repository
BASE_DIR = Path(__file__).parent.parent


class TestManifestValidation:
    """Test suite for validating all manifest files."""

    # Constants
    EXPECTED_PROJECT_NAME = "nethical"

    @staticmethod
    def _check_api_config(api_data, require_url=True):
        """Helper method to validate API configuration."""
        assert "type" in api_data
        if require_url:
            # Check for either 'url' or 'base_url' field
            assert "url" in api_data or "base_url" in api_data

    def test_ai_plugin_json_valid(self):
        """Test that ai-plugin.json is valid JSON."""
        manifest_path = BASE_DIR / "ai-plugin.json"
        assert manifest_path.exists(), "ai-plugin.json not found"

        with open(manifest_path) as f:
            data = json.load(f)

        # Check required fields
        assert "schema_version" in data
        assert "name_for_human" in data
        assert "name_for_model" in data
        assert "description_for_human" in data
        assert "description_for_model" in data
        assert "auth" in data
        assert "api" in data

        # Check API configuration using helper
        self._check_api_config(data["api"])

    def test_grok_manifest_json_valid(self):
        """Test that grok-manifest.json is valid JSON."""
        manifest_path = BASE_DIR / "grok-manifest.json"
        assert manifest_path.exists(), "grok-manifest.json not found"

        with open(manifest_path) as f:
            data = json.load(f)

        # Check required fields
        assert "manifest_version" in data
        assert "name" in data
        assert "display_name" in data
        assert "description" in data
        assert "version" in data
        assert "api" in data

        # Check API configuration using helper (Grok has base_url, not url)
        api = data["api"]
        self._check_api_config(api, require_url=True)
        assert "functions" in api
        assert len(api["functions"]) > 0

        # Check function definition
        func = api["functions"][0]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func

    def test_gemini_manifest_json_valid(self):
        """Test that gemini-manifest.json is valid JSON."""
        manifest_path = BASE_DIR / "gemini-manifest.json"
        assert manifest_path.exists(), "gemini-manifest.json not found"

        with open(manifest_path) as f:
            data = json.load(f)

        # Check required fields
        assert "manifest_version" in data
        assert "name" in data
        assert "display_name" in data
        assert "version" in data
        assert "gemini_integration" in data

        # Check Gemini-specific fields
        integration = data["gemini_integration"]
        assert "type" in integration
        assert "compatible_models" in integration

    def test_langchain_tool_json_valid(self):
        """Test that langchain-tool.json is valid JSON."""
        manifest_path = BASE_DIR / "langchain-tool.json"
        assert manifest_path.exists(), "langchain-tool.json not found"

        with open(manifest_path) as f:
            data = json.load(f)

        # Check required fields
        assert "name" in data
        assert "display_name" in data
        assert "version" in data
        assert "tool_type" in data
        assert "langchain_integration" in data

        # Check LangChain-specific fields
        integration = data["langchain_integration"]
        assert "type" in integration
        assert "base_class" in integration

    def test_autogen_manifest_json_valid(self):
        """Test that autogen-manifest.json is valid JSON."""
        manifest_path = BASE_DIR / "autogen-manifest.json"
        assert manifest_path.exists(), "autogen-manifest.json not found"

        with open(manifest_path) as f:
            data = json.load(f)

        # Check required fields
        assert "name" in data
        assert "display_name" in data
        assert "version" in data
        assert "autogen_integration" in data

        # Check AutoGen-specific fields
        integration = data["autogen_integration"]
        assert "type" in integration
        assert "compatible_versions" in integration

    def test_huggingface_tool_yaml_valid(self):
        """Test that huggingface-tool.yaml is valid YAML."""
        manifest_path = BASE_DIR / "huggingface-tool.yaml"
        assert manifest_path.exists(), "huggingface-tool.yaml not found"

        with open(manifest_path) as f:
            data = yaml.safe_load(f)

        # Check required fields
        assert "name" in data
        assert "display_name" in data
        assert "version" in data
        assert "huggingface_integration" in data

        # Check HuggingFace-specific fields
        integration = data["huggingface_integration"]
        assert "type" in integration
        assert "compatible_with" in integration

    def test_mlflow_integration_yaml_valid(self):
        """Test that mlflow-integration.yaml is valid YAML."""
        manifest_path = BASE_DIR / "mlflow-integration.yaml"
        assert manifest_path.exists(), "mlflow-integration.yaml not found"

        with open(manifest_path) as f:
            data = yaml.safe_load(f)

        # Check required fields
        assert "name" in data
        assert "display_name" in data
        assert "version" in data
        assert "mlflow_integration" in data

        # Check MLflow-specific fields
        integration = data["mlflow_integration"]
        assert "type" in integration
        assert "compatible_versions" in integration

    def test_enterprise_mcp_yaml_valid(self):
        """Test that enterprise-mcp-integrations.yaml is valid YAML."""
        manifest_path = BASE_DIR / "enterprise-mcp-integrations.yaml"
        assert manifest_path.exists(), "enterprise-mcp-integrations.yaml not found"

        with open(manifest_path) as f:
            data = yaml.safe_load(f)

        # Check required fields
        assert "integrations" in data
        integrations = data["integrations"]

        # Check that key platforms are present
        assert "sagemaker" in integrations
        assert "azureml" in integrations
        assert "wandb" in integrations
        assert "vertex_ai" in integrations
        assert "databricks" in integrations

        # Check each integration has required fields
        for name, integration in integrations.items():
            assert "name" in integration, f"{name} missing 'name'"
            assert "version" in integration, f"{name} missing 'version'"
            assert "status" in integration, f"{name} missing 'status'"
            assert "description" in integration, f"{name} missing 'description'"

    def test_openapi_yaml_valid(self):
        """Test that openapi.yaml is valid YAML and has required fields."""
        manifest_path = BASE_DIR / "openapi.yaml"
        assert manifest_path.exists(), "openapi.yaml not found"

        with open(manifest_path) as f:
            data = yaml.safe_load(f)

        # Check OpenAPI required fields
        assert "openapi" in data
        assert data["openapi"].startswith("3.")  # OpenAPI 3.x
        assert "info" in data
        assert "paths" in data

        # Check info section
        info = data["info"]
        assert "title" in info
        assert "version" in info
        assert "description" in info

        # Check paths
        paths = data["paths"]
        assert "/evaluate" in paths
        assert "/health" in paths

        # Check evaluate endpoint
        evaluate = paths["/evaluate"]
        assert "post" in evaluate

        post = evaluate["post"]
        assert "requestBody" in post
        assert "responses" in post

    def test_all_manifests_have_version(self):
        """Test that all manifest files have version field."""
        manifest_files = [
            # ai-plugin.json follows OpenAI spec which doesn't require version
            (BASE_DIR / "grok-manifest.json", json.load),
            (BASE_DIR / "gemini-manifest.json", json.load),
            (BASE_DIR / "langchain-tool.json", json.load),
            (BASE_DIR / "autogen-manifest.json", json.load),
            (BASE_DIR / "huggingface-tool.yaml", yaml.safe_load),
            (BASE_DIR / "mlflow-integration.yaml", yaml.safe_load),
        ]

        for path, loader in manifest_files:
            if path.exists():
                with open(path) as f:
                    data = loader(f)
                assert "version" in data, f"{path.name} missing version"

    def test_all_manifests_have_description(self):
        """Test that all manifest files have description field."""
        manifest_files = [
            (BASE_DIR / "grok-manifest.json", json.load),
            (BASE_DIR / "gemini-manifest.json", json.load),
            (BASE_DIR / "langchain-tool.json", json.load),
            (BASE_DIR / "autogen-manifest.json", json.load),
            (BASE_DIR / "huggingface-tool.yaml", yaml.safe_load),
            (BASE_DIR / "mlflow-integration.yaml", yaml.safe_load),
        ]

        for path, loader in manifest_files:
            if path.exists():
                with open(path) as f:
                    data = loader(f)
                assert "description" in data, f"{path.name} missing description"

    def test_consistent_naming(self):
        """Test that manifest names are consistent."""
        manifest_files = [
            (BASE_DIR / "grok-manifest.json", json.load),
            (BASE_DIR / "gemini-manifest.json", json.load),
            (BASE_DIR / "langchain-tool.json", json.load),
            (BASE_DIR / "autogen-manifest.json", json.load),
            (BASE_DIR / "huggingface-tool.yaml", yaml.safe_load),
            (BASE_DIR / "mlflow-integration.yaml", yaml.safe_load),
        ]

        for path, loader in manifest_files:
            if path.exists():
                with open(path) as f:
                    data = loader(f)
                assert (
                    data.get("name") == self.EXPECTED_PROJECT_NAME
                ), f"{path.name} has inconsistent name"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
