"""
Tests for External Integrations: ML Platform Integrations

Tests the ML platform integration interfaces including MLflow, W&B, and SageMaker.
"""

import pytest
from datetime import datetime

from nethical.integrations.ml_platforms import (
    MLPlatform,
    ExperimentRun,
    MLPlatformInterface,
    MLflowIntegration,
    WandBIntegration,
    SageMakerIntegration,
    MLPlatformManager,
    RunStatus
)


class TestExperimentRun:
    """Test ExperimentRun dataclass"""
    
    def test_experiment_run_creation(self):
        """Test creating an experiment run"""
        run = ExperimentRun(
            run_id="run_123",
            experiment_name="test_experiment",
            parameters={"lr": 0.001, "batch_size": 32},
            metrics={"accuracy": 0.95, "loss": 0.05}
        )
        
        assert run.run_id == "run_123"
        assert run.experiment_name == "test_experiment"
        assert run.parameters["lr"] == 0.001
        assert run.metrics["accuracy"] == 0.95
        assert run.status == RunStatus.RUNNING
    
    def test_experiment_run_to_dict(self):
        """Test converting experiment run to dict"""
        run = ExperimentRun(
            run_id="run_123",
            experiment_name="test_experiment",
            parameters={"lr": 0.001},
            metrics={"accuracy": 0.95}
        )
        
        run_dict = run.to_dict()
        assert run_dict['run_id'] == "run_123"
        assert run_dict['experiment_name'] == "test_experiment"
        assert run_dict['parameters']['lr'] == 0.001
        assert run_dict['metrics']['accuracy'] == 0.95
        assert 'start_time' in run_dict


class TestMLflowIntegration:
    """Test MLflow integration (stub)"""
    
    def test_integration_creation(self):
        """Test creating MLflow integration"""
        integration = MLflowIntegration(tracking_uri="http://localhost:5000")
        
        assert integration.tracking_uri == "http://localhost:5000"
        assert len(integration.active_runs) == 0
    
    def test_start_run(self):
        """Test starting an MLflow run"""
        integration = MLflowIntegration()
        
        run_id = integration.start_run(
            experiment_name="test_experiment",
            run_name="test_run"
        )
        
        assert run_id is not None
        assert run_id in integration.active_runs
        
        run = integration.active_runs[run_id]
        assert run.experiment_name == "test_experiment"
        assert run.tags.get('run_name') == "test_run"
    
    def test_log_parameters(self):
        """Test logging parameters"""
        integration = MLflowIntegration()
        run_id = integration.start_run("test_experiment")
        
        params = {"learning_rate": 0.001, "batch_size": 32, "epochs": 10}
        integration.log_parameters(run_id, params)
        
        run = integration.active_runs[run_id]
        assert run.parameters == params
    
    def test_log_metrics(self):
        """Test logging metrics"""
        integration = MLflowIntegration()
        run_id = integration.start_run("test_experiment")
        
        metrics = {"accuracy": 0.95, "loss": 0.05, "f1_score": 0.93}
        integration.log_metrics(run_id, metrics)
        
        run = integration.active_runs[run_id]
        assert run.metrics == metrics
    
    def test_log_metrics_with_step(self):
        """Test logging metrics with step"""
        integration = MLflowIntegration()
        run_id = integration.start_run("test_experiment")
        
        # Log metrics at different steps
        integration.log_metrics(run_id, {"loss": 0.5}, step=0)
        integration.log_metrics(run_id, {"loss": 0.3}, step=1)
        integration.log_metrics(run_id, {"loss": 0.1}, step=2)
        
        run = integration.active_runs[run_id]
        # Stub implementation keeps only the latest value
        assert run.metrics["loss"] == 0.1
    
    def test_log_artifact(self):
        """Test logging artifacts"""
        integration = MLflowIntegration()
        run_id = integration.start_run("test_experiment")
        
        integration.log_artifact(run_id, "/path/to/model.pkl")
        integration.log_artifact(run_id, "/path/to/plot.png")
        
        run = integration.active_runs[run_id]
        assert len(run.artifacts) == 2
        assert "/path/to/model.pkl" in run.artifacts
        assert "/path/to/plot.png" in run.artifacts
    
    def test_end_run(self):
        """Test ending a run"""
        integration = MLflowIntegration()
        run_id = integration.start_run("test_experiment")
        
        integration.end_run(run_id, status="completed")
        
        run = integration.active_runs[run_id]
        assert run.status == RunStatus.COMPLETED
        assert run.end_time is not None
    
    def test_complete_workflow(self):
        """Test complete MLflow workflow"""
        integration = MLflowIntegration()
        
        # Start run
        run_id = integration.start_run("test_experiment", "workflow_run")
        
        # Log parameters
        integration.log_parameters(run_id, {
            "model": "random_forest",
            "n_estimators": 100
        })
        
        # Log metrics
        integration.log_metrics(run_id, {
            "train_accuracy": 0.98,
            "val_accuracy": 0.95
        })
        
        # Log artifact
        integration.log_artifact(run_id, "model.pkl")
        
        # End run
        integration.end_run(run_id, "completed")
        
        # Verify
        run = integration.active_runs[run_id]
        assert run.status == RunStatus.COMPLETED
        assert run.parameters["model"] == "random_forest"
        assert run.metrics["train_accuracy"] == 0.98
        assert "model.pkl" in run.artifacts


class TestWandBIntegration:
    """Test Weights & Biases integration (stub)"""
    
    def test_integration_creation(self):
        """Test creating W&B integration"""
        integration = WandBIntegration(project="test-project", entity="test-team")
        
        assert integration.project == "test-project"
        assert integration.entity == "test-team"
    
    def test_start_run(self):
        """Test starting a W&B run"""
        integration = WandBIntegration(project="test-project")
        
        run_id = integration.start_run(
            experiment_name="test_experiment",
            run_name="test_run"
        )
        
        assert run_id is not None
        assert run_id in integration.active_runs
    
    def test_log_parameters(self):
        """Test logging config to W&B"""
        integration = WandBIntegration(project="test-project")
        run_id = integration.start_run("test_experiment")
        
        config = {"lr": 0.001, "optimizer": "adam"}
        integration.log_parameters(run_id, config)
        
        run = integration.active_runs[run_id]
        assert run.parameters == config
    
    def test_log_metrics(self):
        """Test logging metrics to W&B"""
        integration = WandBIntegration(project="test-project")
        run_id = integration.start_run("test_experiment")
        
        metrics = {"loss": 0.05, "accuracy": 0.95}
        integration.log_metrics(run_id, metrics)
        
        run = integration.active_runs[run_id]
        assert run.metrics == metrics


class TestSageMakerIntegration:
    """Test SageMaker integration (stub)"""
    
    def test_integration_creation(self):
        """Test creating SageMaker integration"""
        integration = SageMakerIntegration(region="us-west-2", role="arn:aws:iam::...")
        
        assert integration.region == "us-west-2"
        assert integration.role == "arn:aws:iam::..."
    
    def test_start_run(self):
        """Test starting a SageMaker training job"""
        integration = SageMakerIntegration()
        
        run_id = integration.start_run("test_experiment")
        
        assert run_id is not None
        assert run_id.startswith("sm-")
        assert run_id in integration.active_runs
    
    def test_log_parameters(self):
        """Test logging hyperparameters"""
        integration = SageMakerIntegration()
        run_id = integration.start_run("test_experiment")
        
        hyperparams = {"max_depth": 10, "min_child_weight": 3}
        integration.log_parameters(run_id, hyperparams)
        
        run = integration.active_runs[run_id]
        assert run.parameters == hyperparams


class TestMLPlatformManager:
    """Test ML platform manager"""
    
    def test_manager_creation(self):
        """Test creating ML platform manager"""
        manager = MLPlatformManager()
        assert len(manager.platforms) == 0
    
    def test_add_platform(self):
        """Test adding platforms"""
        manager = MLPlatformManager()
        
        mlflow = MLflowIntegration()
        wandb = WandBIntegration("test-project")
        
        manager.add_platform("mlflow", mlflow)
        manager.add_platform("wandb", wandb)
        
        assert len(manager.platforms) == 2
        assert "mlflow" in manager.platforms
        assert "wandb" in manager.platforms
    
    def test_start_run_all(self):
        """Test starting runs on all platforms"""
        manager = MLPlatformManager()
        
        manager.add_platform("mlflow", MLflowIntegration())
        manager.add_platform("wandb", WandBIntegration("test-project"))
        manager.add_platform("sagemaker", SageMakerIntegration())
        
        run_ids = manager.start_run_all("test_experiment", "test_run")
        
        assert len(run_ids) == 3
        assert "mlflow" in run_ids
        assert "wandb" in run_ids
        assert "sagemaker" in run_ids
        
        # Verify each platform has the run
        for platform_name, run_id in run_ids.items():
            platform = manager.platforms[platform_name]
            assert run_id in platform.active_runs
    
    def test_log_metrics_all(self):
        """Test logging metrics to all platforms"""
        manager = MLPlatformManager()
        
        manager.add_platform("mlflow", MLflowIntegration())
        manager.add_platform("wandb", WandBIntegration("test-project"))
        
        run_ids = manager.start_run_all("test_experiment")
        
        metrics = {"accuracy": 0.95, "loss": 0.05}
        manager.log_metrics_all(run_ids, metrics)
        
        # Verify each platform received the metrics
        for platform_name, run_id in run_ids.items():
            platform = manager.platforms[platform_name]
            run = platform.active_runs[run_id]
            assert run.metrics == metrics
    
    def test_end_run_all(self):
        """Test ending runs on all platforms"""
        manager = MLPlatformManager()
        
        manager.add_platform("mlflow", MLflowIntegration())
        manager.add_platform("wandb", WandBIntegration("test-project"))
        
        run_ids = manager.start_run_all("test_experiment")
        manager.end_run_all(run_ids, status="completed")
        
        # Verify each platform ended the run
        for platform_name, run_id in run_ids.items():
            platform = manager.platforms[platform_name]
            run = platform.active_runs[run_id]
            assert run.status == RunStatus.COMPLETED
    
    def test_complete_multi_platform_workflow(self):
        """Test complete workflow across multiple platforms"""
        manager = MLPlatformManager()
        
        # Setup platforms
        manager.add_platform("mlflow", MLflowIntegration("http://localhost:5000"))
        manager.add_platform("wandb", WandBIntegration("my-project", "my-team"))
        
        # Start runs
        run_ids = manager.start_run_all("classification_experiment", "run_001")
        
        # Log parameters
        params = {
            "model": "xgboost",
            "max_depth": 10,
            "learning_rate": 0.1
        }
        for platform_name, run_id in run_ids.items():
            manager.platforms[platform_name].log_parameters(run_id, params)
        
        # Log metrics
        manager.log_metrics_all(run_ids, {
            "accuracy": 0.96,
            "precision": 0.94,
            "recall": 0.95
        })
        
        # End runs
        manager.end_run_all(run_ids, "completed")
        
        # Verify all platforms have complete data
        for platform_name, run_id in run_ids.items():
            platform = manager.platforms[platform_name]
            run = platform.active_runs[run_id]
            
            assert run.status == RunStatus.COMPLETED
            assert run.parameters["model"] == "xgboost"
            assert run.metrics["accuracy"] == 0.96
    
    def test_error_handling(self):
        """Test error handling when platform fails"""
        manager = MLPlatformManager()
        
        # Add a platform that will fail
        class FailingPlatform(MLPlatformInterface):
            def start_run(self, experiment_name, run_name=None):
                raise Exception("Simulated failure")
            def log_parameters(self, run_id, parameters):
                pass
            def log_metrics(self, run_id, metrics, step=None):
                pass
            def log_artifact(self, run_id, artifact_path):
                pass
            def end_run(self, run_id, status="completed"):
                pass
        
        manager.add_platform("failing", FailingPlatform())
        manager.add_platform("working", MLflowIntegration())
        
        # Start runs - failing platform should not prevent working platform
        run_ids = manager.start_run_all("test_experiment")
        
        # Working platform should have succeeded
        assert "working" in run_ids
        # Failing platform should not be in run_ids
        assert "failing" not in run_ids


class TestIntegration:
    """Integration tests for ML platforms"""
    
    def test_multi_experiment_tracking(self):
        """Test tracking multiple experiments across platforms"""
        manager = MLPlatformManager()
        manager.add_platform("mlflow", MLflowIntegration())
        manager.add_platform("wandb", WandBIntegration("test-project"))
        
        experiments = [
            ("experiment_1", {"lr": 0.001}, {"accuracy": 0.90}),
            ("experiment_2", {"lr": 0.01}, {"accuracy": 0.92}),
            ("experiment_3", {"lr": 0.1}, {"accuracy": 0.88})
        ]
        
        for exp_name, params, metrics in experiments:
            run_ids = manager.start_run_all(exp_name)
            
            for platform_name, run_id in run_ids.items():
                platform = manager.platforms[platform_name]
                platform.log_parameters(run_id, params)
                platform.log_metrics(run_id, metrics)
            
            manager.end_run_all(run_ids, "completed")
        
        # Verify all experiments tracked
        mlflow = manager.platforms["mlflow"]
        assert len(mlflow.active_runs) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
