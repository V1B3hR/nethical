"""
MLOps Data Pipeline Module

This module provides comprehensive data ingestion, validation, and preprocessing
functionality for machine learning operations in the Nethical project.

Features:
- Data ingestion from multiple sources (local, Kaggle, S3, etc.)
- Data validation and quality checks
- Data versioning and lineage tracking
- Schema validation
- Data preprocessing and transformation
"""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd


# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """Supported data sources"""
    LOCAL = "local"
    KAGGLE = "kaggle"
    S3 = "s3"
    HTTP = "http"
    DATABASE = "database"


class DataStatus(Enum):
    """Data validation status"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"
    PENDING = "pending"


@dataclass
class DataSchema:
    """Data schema definition for validation"""
    name: str
    version: str
    columns: Dict[str, str]  # column_name -> data_type
    required_columns: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self, df: pd.DataFrame) -> tuple[bool, List[str]]:
        """Validate dataframe against schema"""
        errors = []
        
        # Check required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check column types
        for col, expected_type in self.columns.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if not self._type_compatible(actual_type, expected_type):
                    errors.append(f"Column {col}: expected {expected_type}, got {actual_type}")
        
        # Check constraints
        for col, constraints in self.constraints.items():
            if col in df.columns:
                if 'min' in constraints and df[col].min() < constraints['min']:
                    errors.append(f"Column {col}: value below minimum {constraints['min']}")
                if 'max' in constraints and df[col].max() > constraints['max']:
                    errors.append(f"Column {col}: value above maximum {constraints['max']}")
        
        return len(errors) == 0, errors
    
    def _type_compatible(self, actual: str, expected: str) -> bool:
        """Check if types are compatible"""
        type_map = {
            'int64': ['int', 'integer', 'int64'],
            'float64': ['float', 'float64', 'double'],
            'object': ['str', 'string', 'object'],
            'bool': ['bool', 'boolean'],
        }
        
        for k, v in type_map.items():
            if k in actual and expected in v:
                return True
        return False


@dataclass
class DataVersion:
    """Data version metadata"""
    version_id: str
    source: DataSource
    path: str
    schema: DataSchema
    checksum: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: DataStatus = DataStatus.PENDING
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'version_id': self.version_id,
            'source': self.source.value,
            'path': self.path,
            'schema': {
                'name': self.schema.name,
                'version': self.schema.version
            },
            'checksum': self.checksum,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata,
            'status': self.status.value,
            'validation_errors': self.validation_errors
        }


class DataPipeline:
    """Comprehensive data pipeline for ML operations"""
    
    def __init__(self, workspace_dir: Union[str, Path] = "data"):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        self.raw_dir = self.workspace / "raw"
        self.processed_dir = self.workspace / "processed"
        self.versions_dir = self.workspace / "versions"
        
        for d in [self.raw_dir, self.processed_dir, self.versions_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.versions: Dict[str, DataVersion] = {}
        self._load_versions()
        
    def ingest(self, 
               source: Union[str, Path],
               source_type: DataSource = DataSource.LOCAL,
               schema: Optional[DataSchema] = None,
               validate: bool = True) -> DataVersion:
        """
        Ingest data from source with validation
        
        Args:
            source: Path or URL to data source
            source_type: Type of data source
            schema: Optional schema for validation
            validate: Whether to validate data
            
        Returns:
            DataVersion object with metadata
        """
        logger.info(f"Ingesting data from {source} (type: {source_type.value})")
        
        # Read data based on source type
        if source_type == DataSource.LOCAL:
            df = self._read_local(source)
        elif source_type == DataSource.KAGGLE:
            df = self._read_kaggle(source)
        elif source_type == DataSource.HTTP:
            df = self._read_http(source)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
        
        # Generate version ID and checksum
        checksum = self._compute_checksum(df)
        version_id = f"v_{checksum[:12]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Save raw data
        output_path = self.raw_dir / f"{version_id}.parquet"
        df.to_parquet(output_path)
        logger.info(f"Saved raw data to {output_path}")
        
        # Create version object
        if schema is None:
            schema = self._infer_schema(df)
        
        version = DataVersion(
            version_id=version_id,
            source=source_type,
            path=str(output_path),
            schema=schema,
            checksum=checksum,
            metadata={
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns)
            }
        )
        
        # Validate if requested
        if validate and schema:
            is_valid, errors = schema.validate(df)
            version.status = DataStatus.VALID if is_valid else DataStatus.INVALID
            version.validation_errors = errors
            
            if is_valid:
                logger.info(f"Data validation passed for {version_id}")
            else:
                logger.warning(f"Data validation failed for {version_id}: {errors}")
        else:
            version.status = DataStatus.WARNING
        
        # Store version
        self.versions[version_id] = version
        self._save_version(version)
        
        return version
    
    def validate_data(self, version_id: str, schema: Optional[DataSchema] = None) -> tuple[bool, List[str]]:
        """Validate a data version against schema"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = self.versions[version_id]
        df = pd.read_parquet(version.path)
        
        if schema is None:
            schema = version.schema
        
        is_valid, errors = schema.validate(df)
        version.status = DataStatus.VALID if is_valid else DataStatus.INVALID
        version.validation_errors = errors
        self._save_version(version)
        
        return is_valid, errors
    
    def preprocess(self, 
                   version_id: str,
                   transformations: Optional[List[callable]] = None,
                   save: bool = True) -> pd.DataFrame:
        """
        Apply preprocessing transformations to data
        
        Args:
            version_id: Version ID to process
            transformations: List of transformation functions
            save: Whether to save processed data
            
        Returns:
            Processed dataframe
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = self.versions[version_id]
        df = pd.read_parquet(version.path)
        
        logger.info(f"Preprocessing {version_id}")
        
        if transformations:
            for transform in transformations:
                df = transform(df)
                logger.info(f"Applied transformation: {transform.__name__}")
        
        if save:
            output_path = self.processed_dir / f"{version_id}_processed.parquet"
            df.to_parquet(output_path)
            logger.info(f"Saved processed data to {output_path}")
        
        return df
    
    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """Get version metadata"""
        return self.versions.get(version_id)
    
    def list_versions(self, status: Optional[DataStatus] = None) -> List[DataVersion]:
        """List all versions, optionally filtered by status"""
        versions = list(self.versions.values())
        if status:
            versions = [v for v in versions if v.status == status]
        return sorted(versions, key=lambda v: v.timestamp, reverse=True)
    
    def _read_local(self, path: Union[str, Path]) -> pd.DataFrame:
        """Read data from local file"""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        
        if path.suffix == '.csv':
            return pd.read_csv(path)
        elif path.suffix in ['.parquet', '.pq']:
            return pd.read_parquet(path)
        elif path.suffix == '.json':
            return pd.read_json(path)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def _read_kaggle(self, dataset_name: str) -> pd.DataFrame:
        """Read data from Kaggle (placeholder)"""
        logger.warning(f"Kaggle API integration not implemented. Manual download required: {dataset_name}")
        # TODO: Integrate with Kaggle API
        raise NotImplementedError("Kaggle API integration coming soon")
    
    def _read_http(self, url: str) -> pd.DataFrame:
        """Read data from HTTP URL"""
        logger.info(f"Fetching data from {url}")
        return pd.read_csv(url)
    
    def _compute_checksum(self, df: pd.DataFrame) -> str:
        """Compute checksum for dataframe"""
        data_str = df.to_json(orient='records')
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def _infer_schema(self, df: pd.DataFrame) -> DataSchema:
        """Infer schema from dataframe"""
        columns = {col: str(df[col].dtype) for col in df.columns}
        return DataSchema(
            name="auto_inferred",
            version="1.0",
            columns=columns,
            required_columns=list(df.columns)
        )
    
    def _save_version(self, version: DataVersion):
        """Save version metadata"""
        version_file = self.versions_dir / f"{version.version_id}.json"
        with open(version_file, 'w') as f:
            json.dump(version.to_dict(), f, indent=2)
    
    def _load_versions(self):
        """Load existing versions"""
        for version_file in self.versions_dir.glob("*.json"):
            try:
                with open(version_file) as f:
                    data = json.load(f)
                    # Reconstruct version object (simplified)
                    # Note: Full reconstruction would need to deserialize schema
                    logger.debug(f"Loaded version: {data['version_id']}")
            except Exception as e:
                logger.error(f"Failed to load version {version_file}: {e}")


# Legacy functions for backward compatibility
def fetch_kaggle_dataset(url):
    """Legacy function - use DataPipeline.ingest() instead"""
    logger.info(f"Manual download required. Please download the dataset from: {url}")
    # TODO: Integrate with Kaggle API for automated downloads


def ingest_all(dataset_list_path=Path("datasets/datasets"), download_dir=Path("data/external")):
    """Legacy function - use DataPipeline class instead"""
    download_dir.mkdir(parents=True, exist_ok=True)
    try:
        with open(dataset_list_path) as f:
            for line in f:
                url = line.strip()
                if url and url.startswith("http"):
                    fetch_kaggle_dataset(url)
        logger.info(f"Dataset ingestion completed. Please check {download_dir}/ directory.")
    except FileNotFoundError:
        logger.error(f"Dataset list file not found: {dataset_list_path}")
    except Exception as e:
        logger.error(f"An error occurred during ingestion: {e}")


if __name__ == "__main__":
    # Demo usage
    pipeline = DataPipeline()
    print("Data pipeline initialized")
    print(f"Workspace: {pipeline.workspace}")
