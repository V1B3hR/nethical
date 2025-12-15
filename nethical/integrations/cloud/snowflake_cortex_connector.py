"""Snowflake Cortex integration with Nethical governance."""

from typing import Dict, Any, Optional, List
import logging

from .base import CloudMLProvider, ExperimentRun, RunStatus

logger = logging.getLogger(__name__)


class SnowflakeCortexConnector(CloudMLProvider):
    """Snowflake Cortex integration with Nethical governance."""
    
    def __init__(
        self,
        account: str,
        user: str,
        password: Optional[str] = None,
        warehouse: str = "COMPUTE_WH",
        database: str = "GOVERNANCE_DB",
        schema: str = "PUBLIC",
        enable_governance: bool = True
    ):
        """Initialize Snowflake Cortex connector.
        
        Args:
            account: Snowflake account identifier
            user: Snowflake username
            password: Snowflake password (optional, can use env var)
            warehouse: Warehouse name
            database: Database name
            schema: Schema name
            enable_governance: Whether to enable governance checks
        """
        self.account = account
        self.user = user
        self.warehouse = warehouse
        self.database = database
        self.schema = schema
        self.enable_governance = enable_governance
        self.active_runs: Dict[str, ExperimentRun] = {}
        self.connection = None
        self.available = False
        
        try:
            import snowflake.connector
            
            self.connection = snowflake.connector.connect(
                account=account,
                user=user,
                password=password,
                warehouse=warehouse,
                database=database,
                schema=schema
            )
            self.available = True
            logger.info(f"Snowflake Cortex connector initialized for account {account}")
        except ImportError:
            logger.warning("Snowflake connector not available. Install with: pip install snowflake-connector-python")
        except Exception as e:
            logger.error(f"Failed to initialize Snowflake: {e}")
        
        if enable_governance:
            try:
                from nethical.core import IntegratedGovernance
                self.governance = IntegratedGovernance()
                logger.info("Governance enabled for Snowflake Cortex")
            except Exception as e:
                logger.warning(f"Could not initialize governance: {e}")
                self.enable_governance = False
    
    def start_run(self, experiment_name: str, run_name: Optional[str] = None) -> str:
        """Start a Snowflake Cortex experiment run."""
        if not self.available:
            logger.error("Snowflake not available")
            return "unavailable"
        
        try:
            import uuid
            from datetime import datetime
            
            run_id = str(uuid.uuid4())
            run_name = run_name or f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create experiment tracking table if not exists
            cursor = self.connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS EXPERIMENT_RUNS (
                    run_id VARCHAR(255) PRIMARY KEY,
                    experiment_name VARCHAR(255),
                    run_name VARCHAR(255),
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    status VARCHAR(50),
                    parameters VARIANT,
                    metrics VARIANT
                )
            """)
            
            # Insert new run
            cursor.execute("""
                INSERT INTO EXPERIMENT_RUNS (
                    run_id, experiment_name, run_name, start_time, status, parameters, metrics
                ) VALUES (?, ?, ?, CURRENT_TIMESTAMP(), 'RUNNING', PARSE_JSON('{}'), PARSE_JSON('{}'))
            """, (run_id, experiment_name, run_name))
            
            self.connection.commit()
            cursor.close()
            
            self.active_runs[run_id] = ExperimentRun(
                run_id=run_id,
                experiment_name=experiment_name,
                parameters={},
                metrics={}
            )
            
            logger.info(f"Started Snowflake run: {run_id}")
            return run_id
            
        except Exception as e:
            logger.error(f"Failed to start Snowflake run: {e}")
            return f"error-{e}"
    
    def log_parameters(self, run_id: str, parameters: Dict[str, Any]):
        """Log parameters to Snowflake."""
        if not self.available:
            return
        
        if run_id in self.active_runs:
            self.active_runs[run_id].parameters.update(parameters)
            
            try:
                import json
                cursor = self.connection.cursor()
                cursor.execute("""
                    UPDATE EXPERIMENT_RUNS 
                    SET parameters = PARSE_JSON(?)
                    WHERE run_id = ?
                """, (json.dumps(parameters), run_id))
                self.connection.commit()
                cursor.close()
                logger.debug(f"Logged parameters to Snowflake run {run_id}")
            except Exception as e:
                logger.error(f"Failed to log parameters: {e}")
    
    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to Snowflake."""
        if not self.available:
            return
        
        if run_id in self.active_runs:
            self.active_runs[run_id].metrics.update(metrics)
            
            try:
                import json
                cursor = self.connection.cursor()
                cursor.execute("""
                    UPDATE EXPERIMENT_RUNS 
                    SET metrics = PARSE_JSON(?)
                    WHERE run_id = ?
                """, (json.dumps(metrics), run_id))
                self.connection.commit()
                cursor.close()
                logger.debug(f"Logged metrics to Snowflake run {run_id}")
            except Exception as e:
                logger.error(f"Failed to log metrics: {e}")
    
    def complete_with_governance(
        self,
        model_name: str,
        prompt: str,
        options: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate completion with Snowflake Cortex LLM and governance.
        
        Args:
            model_name: Cortex LLM model name (e.g., 'snowflake-arctic', 'mistral-large')
            prompt: Input prompt
            options: Optional model parameters
            
        Returns:
            Completion result with governance info
        """
        if not self.available:
            return {"error": "Snowflake not available"}
        
        try:
            # Pre-check with governance
            if self.enable_governance:
                result = self.governance.process_action(
                    action=prompt,
                    agent_id=f"snowflake-cortex-{model_name}",
                    action_type="model_input"
                )
                if result.get('risk_score', 0) > 0.7:
                    return {
                        "error": "Prompt blocked by governance",
                        "reason": result.get('reason'),
                        "governance_result": result
                    }
            
            # Generate completion using Cortex
            cursor = self.connection.cursor()
            query = f"SELECT SNOWFLAKE.CORTEX.COMPLETE('{model_name}', ?)"
            
            cursor.execute(query, (prompt,))
            result = cursor.fetchone()
            cursor.close()
            
            completion = result[0] if result else ""
            
            # Post-check with governance
            if self.enable_governance:
                gov_result = self.governance.process_action(
                    action=completion,
                    agent_id=f"snowflake-cortex-{model_name}",
                    action_type="model_output"
                )
                if gov_result.get('risk_score', 0) > 0.7:
                    return {
                        "completion": "[FILTERED]",
                        "governance": gov_result
                    }
            
            return {"completion": completion}
            
        except Exception as e:
            logger.error(f"Failed to generate completion: {e}")
            return {"error": str(e)}
    
    def classify_with_governance(
        self,
        text: str,
        categories: List[str]
    ) -> Dict[str, Any]:
        """Classify text with Snowflake Cortex and governance.
        
        Args:
            text: Text to classify
            categories: List of category labels
            
        Returns:
            Classification result with governance info
        """
        if not self.available:
            return {"error": "Snowflake not available"}
        
        try:
            # Pre-check with governance
            if self.enable_governance:
                result = self.governance.process_action(
                    action=text,
                    agent_id="snowflake-cortex-classify",
                    action_type="model_input"
                )
                if result.get('risk_score', 0) > 0.7:
                    return {
                        "error": "Text blocked by governance",
                        "reason": result.get('reason')
                    }
            
            # Classify using Cortex
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT SNOWFLAKE.CORTEX.CLASSIFY_TEXT(?, ?)",
                (text, ','.join(categories))
            )
            result = cursor.fetchone()
            cursor.close()
            
            return {"classification": result[0] if result else None}
            
        except Exception as e:
            logger.error(f"Failed to classify text: {e}")
            return {"error": str(e)}
    
    def end_run(self, run_id: str, status: str = "completed"):
        """End a Snowflake experiment run."""
        if not self.available:
            return
        
        if run_id in self.active_runs:
            self.active_runs[run_id].status = RunStatus(status)
            
            try:
                cursor = self.connection.cursor()
                cursor.execute("""
                    UPDATE EXPERIMENT_RUNS 
                    SET end_time = CURRENT_TIMESTAMP(), status = ?
                    WHERE run_id = ?
                """, (status, run_id))
                self.connection.commit()
                cursor.close()
                logger.info(f"Ended Snowflake run {run_id} with status {status}")
            except Exception as e:
                logger.error(f"Failed to end run: {e}")
    
    def get_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get run metadata."""
        return self.active_runs.get(run_id)
    
    def close(self):
        """Close Snowflake connection."""
        if self.connection:
            try:
                self.connection.close()
                logger.info("Closed Snowflake connection")
            except Exception as e:
                logger.error(f"Failed to close connection: {e}")
