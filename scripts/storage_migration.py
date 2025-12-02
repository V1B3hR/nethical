#!/usr/bin/env python3
"""
Storage Migration Script for Nethical.

This script migrates data from file-based storage to production-grade
storage backends (PostgreSQL and S3).

Usage:
    python scripts/storage_migration.py --source file --target postgres
    python scripts/storage_migration.py --source file --target s3
    python scripts/storage_migration.py --source file --target all

Environment Variables:
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER, POSTGRES_PASSWORD
    S3_ENDPOINT, S3_ACCESS_KEY, S3_SECRET_KEY, S3_REGION
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib
import gzip

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from nethical.storage.postgres_backend import PostgresBackend, PostgresConfig
from nethical.storage.s3_backend import S3Backend, S3Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StorageMigrator:
    """
    Migrates data from file-based storage to production backends.
    
    Supports migration of:
    - Agent data
    - Model registry entries
    - Policy definitions
    - Audit logs
    - Configuration files
    """
    
    def __init__(
        self,
        source_path: str = "data",
        postgres_config: Optional[PostgresConfig] = None,
        s3_config: Optional[S3Config] = None,
        dry_run: bool = False
    ):
        """
        Initialize the migrator.
        
        Args:
            source_path: Path to file-based storage
            postgres_config: PostgreSQL configuration
            s3_config: S3 configuration
            dry_run: If True, don't actually migrate, just report what would be done
        """
        self.source_path = Path(source_path)
        self.dry_run = dry_run
        self.stats = {
            'agents': {'migrated': 0, 'failed': 0, 'skipped': 0},
            'models': {'migrated': 0, 'failed': 0, 'skipped': 0},
            'policies': {'migrated': 0, 'failed': 0, 'skipped': 0},
            'audit_logs': {'migrated': 0, 'failed': 0, 'skipped': 0},
            'artifacts': {'migrated': 0, 'failed': 0, 'skipped': 0}
        }
        
        # Initialize backends
        self.postgres: Optional[PostgresBackend] = None
        self.s3: Optional[S3Backend] = None
        
        if postgres_config:
            try:
                self.postgres = PostgresBackend(postgres_config)
                logger.info("PostgreSQL backend initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize PostgreSQL: {e}")
                
        if s3_config:
            try:
                self.s3 = S3Backend(s3_config)
                logger.info("S3 backend initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize S3: {e}")
    
    def migrate_all(self) -> Dict[str, Any]:
        """
        Run all migrations.
        
        Returns:
            Migration statistics
        """
        logger.info("Starting full migration...")
        
        if self.postgres:
            self.migrate_agents()
            self.migrate_policies()
            self.migrate_audit_logs_to_postgres()
            
        if self.s3:
            self.migrate_models_to_s3()
            self.migrate_artifacts_to_s3()
            self.migrate_audit_logs_to_s3()
            
        self._print_summary()
        return self.stats
    
    def migrate_agents(self) -> None:
        """Migrate agent data to PostgreSQL."""
        if not self.postgres:
            logger.warning("PostgreSQL not available, skipping agent migration")
            return
            
        agents_path = self.source_path / "agents"
        if not agents_path.exists():
            logger.info("No agents directory found, skipping")
            return
            
        logger.info("Migrating agents...")
        
        for agent_file in agents_path.glob("*.json"):
            try:
                with open(agent_file, 'r') as f:
                    agent_data = json.load(f)
                    
                agent_id = agent_data.get('agent_id') or agent_file.stem
                
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would migrate agent: {agent_id}")
                    self.stats['agents']['skipped'] += 1
                    continue
                    
                result = self.postgres.insert_agent(
                    agent_id=agent_id,
                    name=agent_data.get('name'),
                    agent_type=agent_data.get('agent_type', 'general'),
                    trust_level=agent_data.get('trust_level', 0.5),
                    status=agent_data.get('status', 'active'),
                    metadata=agent_data.get('metadata', {}),
                    region_id=agent_data.get('region_id'),
                    logical_domain=agent_data.get('logical_domain', 'default')
                )
                
                if result:
                    logger.debug(f"Migrated agent: {agent_id}")
                    self.stats['agents']['migrated'] += 1
                else:
                    logger.error(f"Failed to migrate agent: {agent_id}")
                    self.stats['agents']['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Error migrating agent {agent_file}: {e}")
                self.stats['agents']['failed'] += 1
    
    def migrate_policies(self) -> None:
        """Migrate policy definitions to PostgreSQL."""
        if not self.postgres:
            logger.warning("PostgreSQL not available, skipping policy migration")
            return
            
        policies_path = self.source_path / "policies"
        if not policies_path.exists():
            logger.info("No policies directory found, skipping")
            return
            
        logger.info("Migrating policies...")
        
        for policy_file in policies_path.glob("**/*.json"):
            try:
                with open(policy_file, 'r') as f:
                    policy_data = json.load(f)
                    
                policy_id = policy_data.get('policy_id') or policy_file.stem
                version = policy_data.get('version', '1.0.0')
                
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would migrate policy: {policy_id} v{version}")
                    self.stats['policies']['skipped'] += 1
                    continue
                    
                result = self.postgres.create_policy_version(
                    policy_id=policy_id,
                    version=version,
                    content=policy_data.get('content', policy_data),
                    policy_type=policy_data.get('policy_type', 'governance'),
                    priority=policy_data.get('priority', 100),
                    created_by=policy_data.get('created_by'),
                    metadata=policy_data.get('metadata', {})
                )
                
                if result:
                    logger.debug(f"Migrated policy: {policy_id} v{version}")
                    self.stats['policies']['migrated'] += 1
                else:
                    logger.error(f"Failed to migrate policy: {policy_id}")
                    self.stats['policies']['failed'] += 1
                    
            except Exception as e:
                logger.error(f"Error migrating policy {policy_file}: {e}")
                self.stats['policies']['failed'] += 1
    
    def migrate_audit_logs_to_postgres(self) -> None:
        """Migrate audit logs to PostgreSQL."""
        if not self.postgres:
            logger.warning("PostgreSQL not available, skipping audit log migration")
            return
            
        audit_path = self.source_path / "audit"
        if not audit_path.exists():
            logger.info("No audit directory found, skipping")
            return
            
        logger.info("Migrating audit logs to PostgreSQL...")
        
        batch = []
        batch_size = 100
        
        for log_file in audit_path.glob("**/*.json"):
            try:
                with open(log_file, 'r') as f:
                    log_data = json.load(f)
                    
                # Handle both single entries and arrays
                entries = log_data if isinstance(log_data, list) else [log_data]
                
                for entry in entries:
                    if self.dry_run:
                        self.stats['audit_logs']['skipped'] += 1
                        continue
                        
                    batch.append({
                        'agent_id': entry.get('agent_id', 'unknown'),
                        'decision': entry.get('decision', 'UNKNOWN'),
                        'action_type': entry.get('action_type'),
                        'action_content': entry.get('action_content') or entry.get('action'),
                        'risk_score': entry.get('risk_score'),
                        'latency_ms': entry.get('latency_ms'),
                        'violations': entry.get('violations', []),
                        'policies_applied': entry.get('policies_applied', []),
                        'context': entry.get('context', {}),
                        'metadata': entry.get('metadata', {}),
                        'region_id': entry.get('region_id'),
                        'logical_domain': entry.get('logical_domain'),
                        'request_id': entry.get('request_id'),
                        'time': self._parse_timestamp(entry.get('timestamp'))
                    })
                    
                    if len(batch) >= batch_size:
                        success, failed = self.postgres.batch_insert_audit_events(batch)
                        self.stats['audit_logs']['migrated'] += success
                        self.stats['audit_logs']['failed'] += failed
                        batch = []
                        
            except Exception as e:
                logger.error(f"Error migrating audit log {log_file}: {e}")
                self.stats['audit_logs']['failed'] += 1
                
        # Insert remaining batch
        if batch and not self.dry_run:
            success, failed = self.postgres.batch_insert_audit_events(batch)
            self.stats['audit_logs']['migrated'] += success
            self.stats['audit_logs']['failed'] += failed
    
    def migrate_models_to_s3(self) -> None:
        """Migrate model artifacts to S3."""
        if not self.s3:
            logger.warning("S3 not available, skipping model migration")
            return
            
        models_path = self.source_path / "models"
        if not models_path.exists():
            logger.info("No models directory found, skipping")
            return
            
        logger.info("Migrating models to S3...")
        
        for model_dir in models_path.iterdir():
            if not model_dir.is_dir():
                continue
                
            model_name = model_dir.name
            
            for version_dir in model_dir.iterdir():
                if not version_dir.is_dir():
                    continue
                    
                version = version_dir.name
                
                # Find model file
                model_file = None
                for ext in ['.bin', '.pt', '.pkl', '.h5', '.onnx', '.safetensors']:
                    candidates = list(version_dir.glob(f"*{ext}"))
                    if candidates:
                        model_file = candidates[0]
                        break
                        
                if not model_file:
                    logger.warning(f"No model file found for {model_name}/{version}")
                    continue
                    
                try:
                    if self.dry_run:
                        logger.info(f"[DRY RUN] Would migrate model: {model_name} v{version}")
                        self.stats['models']['skipped'] += 1
                        continue
                        
                    with open(model_file, 'rb') as f:
                        model_data = f.read()
                        
                    # Load metadata if exists
                    metadata = {}
                    metadata_file = version_dir / "metadata.json"
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            
                    result = self.s3.upload_model(
                        model_name=model_name,
                        version=version,
                        data=model_data,
                        metadata={k: str(v) for k, v in metadata.items()}
                    )
                    
                    if result:
                        logger.debug(f"Migrated model: {model_name} v{version}")
                        self.stats['models']['migrated'] += 1
                    else:
                        logger.error(f"Failed to migrate model: {model_name}/{version}")
                        self.stats['models']['failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error migrating model {model_name}/{version}: {e}")
                    self.stats['models']['failed'] += 1
    
    def migrate_artifacts_to_s3(self) -> None:
        """Migrate generic artifacts to S3."""
        if not self.s3:
            logger.warning("S3 not available, skipping artifact migration")
            return
            
        artifacts_path = self.source_path / "artifacts"
        if not artifacts_path.exists():
            logger.info("No artifacts directory found, skipping")
            return
            
        logger.info("Migrating artifacts to S3...")
        
        for artifact_type_dir in artifacts_path.iterdir():
            if not artifact_type_dir.is_dir():
                continue
                
            artifact_type = artifact_type_dir.name
            
            for artifact_file in artifact_type_dir.glob("**/*"):
                if not artifact_file.is_file():
                    continue
                    
                try:
                    artifact_id = artifact_file.relative_to(artifact_type_dir).as_posix()
                    
                    if self.dry_run:
                        logger.info(f"[DRY RUN] Would migrate artifact: {artifact_type}/{artifact_id}")
                        self.stats['artifacts']['skipped'] += 1
                        continue
                        
                    with open(artifact_file, 'rb') as f:
                        data = f.read()
                        
                    result = self.s3.upload_artifact(
                        artifact_type=artifact_type,
                        artifact_id=hashlib.md5(artifact_id.encode()).hexdigest()[:16],
                        data=data,
                        filename=artifact_file.name
                    )
                    
                    if result:
                        logger.debug(f"Migrated artifact: {artifact_type}/{artifact_id}")
                        self.stats['artifacts']['migrated'] += 1
                    else:
                        logger.error(f"Failed to migrate artifact: {artifact_type}/{artifact_id}")
                        self.stats['artifacts']['failed'] += 1
                        
                except Exception as e:
                    logger.error(f"Error migrating artifact {artifact_file}: {e}")
                    self.stats['artifacts']['failed'] += 1
    
    def migrate_audit_logs_to_s3(self) -> None:
        """Migrate audit logs to S3 for archival."""
        if not self.s3:
            logger.warning("S3 not available, skipping audit log S3 migration")
            return
            
        audit_path = self.source_path / "audit"
        if not audit_path.exists():
            logger.info("No audit directory found, skipping S3 migration")
            return
            
        logger.info("Migrating audit logs to S3...")
        
        # Group logs by date for efficient archiving
        logs_by_date: Dict[str, List[Path]] = {}
        
        for log_file in audit_path.glob("**/*.json"):
            # Try to extract date from path or file modification time
            try:
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime, tz=timezone.utc)
                date_key = mtime.strftime('%Y-%m-%d')
                
                if date_key not in logs_by_date:
                    logs_by_date[date_key] = []
                logs_by_date[date_key].append(log_file)
                
            except Exception as e:
                logger.warning(f"Error processing log file date {log_file}: {e}")
        
        # Archive each day's logs
        for date_str, log_files in logs_by_date.items():
            try:
                if self.dry_run:
                    logger.info(f"[DRY RUN] Would archive {len(log_files)} logs for {date_str}")
                    continue
                    
                # Combine logs for the day
                combined_logs = []
                for log_file in log_files:
                    with open(log_file, 'r') as f:
                        log_data = json.load(f)
                        if isinstance(log_data, list):
                            combined_logs.extend(log_data)
                        else:
                            combined_logs.append(log_data)
                            
                # Compress and upload
                log_json = json.dumps(combined_logs, default=str)
                compressed = gzip.compress(log_json.encode('utf-8'))
                
                date = datetime.strptime(date_str, '%Y-%m-%d').replace(tzinfo=timezone.utc)
                log_id = hashlib.md5(date_str.encode()).hexdigest()[:16]
                
                result = self.s3.upload_audit_log(
                    date=date,
                    log_id=log_id,
                    data=compressed,
                    compressed=True
                )
                
                if result:
                    logger.debug(f"Archived audit logs for {date_str}")
                else:
                    logger.error(f"Failed to archive audit logs for {date_str}")
                    
            except Exception as e:
                logger.error(f"Error archiving logs for {date_str}: {e}")
    
    def _parse_timestamp(self, ts: Optional[str]) -> datetime:
        """Parse a timestamp string to datetime."""
        if not ts:
            return datetime.now(timezone.utc)
            
        try:
            # Try ISO format
            return datetime.fromisoformat(ts.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            return datetime.now(timezone.utc)
    
    def _print_summary(self) -> None:
        """Print migration summary."""
        logger.info("\n" + "=" * 60)
        logger.info("MIGRATION SUMMARY")
        logger.info("=" * 60)
        
        for category, counts in self.stats.items():
            total = counts['migrated'] + counts['failed'] + counts['skipped']
            if total > 0:
                logger.info(
                    f"{category.upper():15} - "
                    f"Migrated: {counts['migrated']:5}, "
                    f"Failed: {counts['failed']:5}, "
                    f"Skipped: {counts['skipped']:5}"
                )
                
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Migrate Nethical data from file storage to production backends"
    )
    parser.add_argument(
        '--source',
        type=str,
        default='data',
        help='Source directory for file-based storage'
    )
    parser.add_argument(
        '--target',
        type=str,
        choices=['postgres', 's3', 'all'],
        default='all',
        help='Target backend(s) to migrate to'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be migrated without actually migrating'
    )
    parser.add_argument(
        '--postgres-host',
        type=str,
        default=os.getenv('POSTGRES_HOST', 'localhost'),
        help='PostgreSQL host'
    )
    parser.add_argument(
        '--postgres-port',
        type=int,
        default=int(os.getenv('POSTGRES_PORT', '5432')),
        help='PostgreSQL port'
    )
    parser.add_argument(
        '--postgres-db',
        type=str,
        default=os.getenv('POSTGRES_DB', 'nethical'),
        help='PostgreSQL database'
    )
    parser.add_argument(
        '--postgres-user',
        type=str,
        default=os.getenv('POSTGRES_USER', 'nethical'),
        help='PostgreSQL user'
    )
    parser.add_argument(
        '--postgres-password',
        type=str,
        default=os.getenv('POSTGRES_PASSWORD', ''),
        help='PostgreSQL password'
    )
    parser.add_argument(
        '--s3-endpoint',
        type=str,
        default=os.getenv('S3_ENDPOINT'),
        help='S3 endpoint URL (for MinIO)'
    )
    parser.add_argument(
        '--s3-access-key',
        type=str,
        default=os.getenv('S3_ACCESS_KEY', ''),
        help='S3 access key'
    )
    parser.add_argument(
        '--s3-secret-key',
        type=str,
        default=os.getenv('S3_SECRET_KEY', ''),
        help='S3 secret key'
    )
    parser.add_argument(
        '--s3-region',
        type=str,
        default=os.getenv('S3_REGION', 'us-east-1'),
        help='S3 region'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        
    # Configure backends based on target
    postgres_config = None
    s3_config = None
    
    if args.target in ['postgres', 'all']:
        postgres_config = PostgresConfig(
            host=args.postgres_host,
            port=args.postgres_port,
            database=args.postgres_db,
            user=args.postgres_user,
            password=args.postgres_password
        )
        
    if args.target in ['s3', 'all']:
        s3_config = S3Config(
            endpoint_url=args.s3_endpoint,
            access_key=args.s3_access_key,
            secret_key=args.s3_secret_key,
            region=args.s3_region
        )
    
    # Run migration
    migrator = StorageMigrator(
        source_path=args.source,
        postgres_config=postgres_config,
        s3_config=s3_config,
        dry_run=args.dry_run
    )
    
    if args.dry_run:
        logger.info("DRY RUN MODE - No actual changes will be made")
        
    migrator.migrate_all()


if __name__ == '__main__':
    main()
