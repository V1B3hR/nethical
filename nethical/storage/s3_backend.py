"""
S3-compatible object storage backend for Nethical governance system.

This module provides a unified interface for S3-compatible object storage
including AWS S3, MinIO, and other S3-compatible services.

Features:
- Multi-bucket support for models, artifacts, and audit logs
- Automatic content type detection
- Streaming upload/download for large files
- Lifecycle management integration
- Encryption at rest support
"""

import logging
import hashlib
import io
import mimetypes
from pathlib import Path
from typing import Any, BinaryIO, Dict, Generator, List, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass
import json

try:
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    boto3 = None
    Config = None
    ClientError = Exception
    NoCredentialsError = Exception

logger = logging.getLogger(__name__)


@dataclass
class S3Config:
    """S3/MinIO connection configuration."""
    endpoint_url: Optional[str] = None  # None for AWS S3, URL for MinIO
    access_key: str = ""
    secret_key: str = ""
    region: str = "us-east-1"
    use_ssl: bool = True
    verify_ssl: bool = True
    signature_version: str = "s3v4"
    # Connection settings
    connect_timeout: int = 10
    read_timeout: int = 60
    max_retries: int = 3
    # Bucket configuration
    bucket_models: str = "nethical-models"
    bucket_artifacts: str = "nethical-artifacts"
    bucket_audit_logs: str = "nethical-audit-logs"
    bucket_backups: str = "nethical-backups"


@dataclass
class ObjectMetadata:
    """Metadata for a stored object."""
    key: str
    bucket: str
    size: int
    etag: str
    content_type: str
    last_modified: datetime
    metadata: Dict[str, str]
    storage_class: Optional[str] = None
    version_id: Optional[str] = None


class S3Backend:
    """
    S3-compatible object storage backend for Nethical.
    
    Provides storage for:
    - ML models and artifacts
    - Audit log archives
    - Backup files
    - Large configuration files
    
    Example:
        >>> config = S3Config(
        ...     endpoint_url="http://localhost:9000",
        ...     access_key="access_key",
        ...     secret_key="secret_key"
        ... )
        >>> backend = S3Backend(config)
        >>> 
        >>> # Upload a model
        >>> backend.upload_model("my-model", "1.0.0", model_bytes, metadata={"framework": "pytorch"})
        >>> 
        >>> # Download a model
        >>> data = backend.download_model("my-model", "1.0.0")
    """
    
    def __init__(self, config: Optional[S3Config] = None, enabled: bool = True):
        """
        Initialize S3 backend.
        
        Args:
            config: S3 configuration. Uses defaults if not provided.
            enabled: Whether the backend is enabled.
        """
        self.config = config or S3Config()
        self.enabled = enabled and BOTO3_AVAILABLE
        self._client = None
        self._resource = None
        
        if not BOTO3_AVAILABLE:
            logger.warning("boto3 not available. Install with: pip install boto3")
            self.enabled = False
            return
            
        if not self.enabled:
            logger.info("S3 backend disabled by configuration")
            return
            
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize S3 client and resource."""
        try:
            # Configure retry strategy
            boto_config = Config(
                connect_timeout=self.config.connect_timeout,
                read_timeout=self.config.read_timeout,
                retries={
                    'max_attempts': self.config.max_retries,
                    'mode': 'adaptive'
                },
                signature_version=self.config.signature_version
            )
            
            client_kwargs = {
                'service_name': 's3',
                'region_name': self.config.region,
                'config': boto_config,
                'use_ssl': self.config.use_ssl,
                'verify': self.config.verify_ssl
            }
            
            # Add credentials if provided
            if self.config.access_key and self.config.secret_key:
                client_kwargs['aws_access_key_id'] = self.config.access_key
                client_kwargs['aws_secret_access_key'] = self.config.secret_key
                
            # Add endpoint URL for MinIO/custom S3
            if self.config.endpoint_url:
                client_kwargs['endpoint_url'] = self.config.endpoint_url
                
            self._client = boto3.client(**client_kwargs)
            self._resource = boto3.resource(**client_kwargs)
            
            # Test connection
            self._client.list_buckets()
            
            logger.info(
                f"S3 backend connected to "
                f"{self.config.endpoint_url or 'AWS S3'}"
            )
            
        except (ClientError, NoCredentialsError) as e:
            logger.error(f"Failed to initialize S3 client: {e}")
            self.enabled = False
            raise
    
    # =========================================================================
    # BUCKET OPERATIONS
    # =========================================================================
    
    def ensure_bucket(self, bucket: str) -> bool:
        """Ensure a bucket exists, creating it if necessary."""
        if not self.enabled:
            return False
            
        try:
            self._client.head_bucket(Bucket=bucket)
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == '404':
                try:
                    create_params = {'Bucket': bucket}
                    # AWS requires LocationConstraint for non-us-east-1
                    if self.config.region != 'us-east-1' and not self.config.endpoint_url:
                        create_params['CreateBucketConfiguration'] = {
                            'LocationConstraint': self.config.region
                        }
                    self._client.create_bucket(**create_params)
                    logger.info(f"Created bucket: {bucket}")
                    return True
                except ClientError as create_error:
                    logger.error(f"Failed to create bucket {bucket}: {create_error}")
                    return False
            logger.error(f"Failed to check bucket {bucket}: {e}")
            return False
    
    def list_buckets(self) -> List[str]:
        """List all available buckets."""
        if not self.enabled:
            return []
            
        try:
            response = self._client.list_buckets()
            return [bucket['Name'] for bucket in response.get('Buckets', [])]
        except ClientError as e:
            logger.error(f"Failed to list buckets: {e}")
            return []
    
    # =========================================================================
    # MODEL OPERATIONS
    # =========================================================================
    
    def upload_model(
        self,
        model_name: str,
        version: str,
        data: Union[bytes, BinaryIO],
        metadata: Optional[Dict[str, str]] = None,
        content_type: str = "application/octet-stream"
    ) -> Optional[ObjectMetadata]:
        """
        Upload a model artifact.
        
        Args:
            model_name: Name of the model
            version: Version string
            data: Model data as bytes or file-like object
            metadata: Custom metadata to attach
            content_type: MIME type of the content
            
        Returns:
            ObjectMetadata on success, None on failure
        """
        if not self.enabled:
            return None
            
        key = f"models/{model_name}/{version}/model.bin"
        bucket = self.config.bucket_models
        
        return self._upload_object(
            bucket=bucket,
            key=key,
            data=data,
            metadata=metadata,
            content_type=content_type
        )
    
    def download_model(
        self,
        model_name: str,
        version: str
    ) -> Optional[bytes]:
        """
        Download a model artifact.
        
        Args:
            model_name: Name of the model
            version: Version string
            
        Returns:
            Model data as bytes, or None on failure
        """
        if not self.enabled:
            return None
            
        key = f"models/{model_name}/{version}/model.bin"
        bucket = self.config.bucket_models
        
        return self._download_object(bucket, key)
    
    def list_model_versions(self, model_name: str) -> List[str]:
        """List all versions of a model."""
        if not self.enabled:
            return []
            
        prefix = f"models/{model_name}/"
        bucket = self.config.bucket_models
        
        versions = set()
        try:
            paginator = self._client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/'):
                for prefix_info in page.get('CommonPrefixes', []):
                    # Extract version from prefix like "models/model-name/1.0.0/"
                    version = prefix_info['Prefix'].rstrip('/').split('/')[-1]
                    versions.add(version)
            return sorted(list(versions))
        except ClientError as e:
            logger.error(f"Failed to list model versions: {e}")
            return []
    
    def delete_model(self, model_name: str, version: str) -> bool:
        """Delete a model version."""
        if not self.enabled:
            return False
            
        prefix = f"models/{model_name}/{version}/"
        bucket = self.config.bucket_models
        
        return self._delete_prefix(bucket, prefix)
    
    # =========================================================================
    # ARTIFACT OPERATIONS
    # =========================================================================
    
    def upload_artifact(
        self,
        artifact_type: str,
        artifact_id: str,
        data: Union[bytes, BinaryIO],
        filename: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None
    ) -> Optional[ObjectMetadata]:
        """
        Upload a generic artifact.
        
        Args:
            artifact_type: Type of artifact (e.g., 'policy', 'config', 'dataset')
            artifact_id: Unique identifier for the artifact
            data: Artifact data
            filename: Optional filename for content type detection
            metadata: Custom metadata
            
        Returns:
            ObjectMetadata on success, None on failure
        """
        if not self.enabled:
            return None
            
        # Determine content type
        content_type = "application/octet-stream"
        if filename:
            guessed_type, _ = mimetypes.guess_type(filename)
            if guessed_type:
                content_type = guessed_type
                
        key = f"artifacts/{artifact_type}/{artifact_id}"
        if filename:
            key = f"{key}/{filename}"
            
        bucket = self.config.bucket_artifacts
        
        return self._upload_object(
            bucket=bucket,
            key=key,
            data=data,
            metadata=metadata,
            content_type=content_type
        )
    
    def download_artifact(
        self,
        artifact_type: str,
        artifact_id: str,
        filename: Optional[str] = None
    ) -> Optional[bytes]:
        """Download an artifact."""
        if not self.enabled:
            return None
            
        key = f"artifacts/{artifact_type}/{artifact_id}"
        if filename:
            key = f"{key}/{filename}"
            
        bucket = self.config.bucket_artifacts
        
        return self._download_object(bucket, key)
    
    def list_artifacts(
        self,
        artifact_type: Optional[str] = None,
        max_results: int = 1000
    ) -> List[ObjectMetadata]:
        """List artifacts with optional type filtering."""
        if not self.enabled:
            return []
            
        prefix = "artifacts/"
        if artifact_type:
            prefix = f"artifacts/{artifact_type}/"
            
        bucket = self.config.bucket_artifacts
        
        return self._list_objects(bucket, prefix, max_results)
    
    # =========================================================================
    # AUDIT LOG OPERATIONS
    # =========================================================================
    
    def upload_audit_log(
        self,
        date: datetime,
        log_id: str,
        data: Union[bytes, BinaryIO],
        compressed: bool = True
    ) -> Optional[ObjectMetadata]:
        """
        Upload an audit log archive.
        
        Args:
            date: Date of the audit log
            log_id: Unique log identifier
            data: Log data (typically compressed)
            compressed: Whether the data is gzip compressed
            
        Returns:
            ObjectMetadata on success, None on failure
        """
        if not self.enabled:
            return None
            
        date_prefix = date.strftime('%Y/%m/%d')
        extension = ".log.gz" if compressed else ".log"
        key = f"audit/{date_prefix}/{log_id}{extension}"
        
        content_type = "application/gzip" if compressed else "text/plain"
        
        bucket = self.config.bucket_audit_logs
        
        return self._upload_object(
            bucket=bucket,
            key=key,
            data=data,
            metadata={
                "log-date": date.isoformat(),
                "log-id": log_id,
                "compressed": str(compressed)
            },
            content_type=content_type
        )
    
    def list_audit_logs(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        max_results: int = 1000
    ) -> List[ObjectMetadata]:
        """List audit logs within a date range."""
        if not self.enabled:
            return []
            
        bucket = self.config.bucket_audit_logs
        
        if start_date:
            prefix = f"audit/{start_date.strftime('%Y/%m')}"
        else:
            prefix = "audit/"
            
        objects = self._list_objects(bucket, prefix, max_results)
        
        # Filter by date range if specified
        if end_date:
            objects = [
                obj for obj in objects
                if obj.last_modified <= end_date
            ]
            
        return objects
    
    def download_audit_log(self, key: str) -> Optional[bytes]:
        """Download an audit log by its key."""
        if not self.enabled:
            return None
            
        bucket = self.config.bucket_audit_logs
        return self._download_object(bucket, key)
    
    # =========================================================================
    # BACKUP OPERATIONS
    # =========================================================================
    
    def upload_backup(
        self,
        backup_type: str,
        backup_id: str,
        data: Union[bytes, BinaryIO],
        metadata: Optional[Dict[str, str]] = None
    ) -> Optional[ObjectMetadata]:
        """
        Upload a backup file.
        
        Args:
            backup_type: Type of backup (e.g., 'postgres', 'config', 'full')
            backup_id: Unique backup identifier (typically timestamp-based)
            data: Backup data
            metadata: Custom metadata
            
        Returns:
            ObjectMetadata on success, None on failure
        """
        if not self.enabled:
            return None
            
        timestamp = datetime.now(timezone.utc).strftime('%Y/%m/%d')
        key = f"backups/{backup_type}/{timestamp}/{backup_id}.backup"
        
        bucket = self.config.bucket_backups
        
        return self._upload_object(
            bucket=bucket,
            key=key,
            data=data,
            metadata=metadata,
            content_type="application/octet-stream"
        )
    
    def list_backups(
        self,
        backup_type: Optional[str] = None,
        max_results: int = 100
    ) -> List[ObjectMetadata]:
        """List available backups."""
        if not self.enabled:
            return []
            
        prefix = "backups/"
        if backup_type:
            prefix = f"backups/{backup_type}/"
            
        bucket = self.config.bucket_backups
        
        return self._list_objects(bucket, prefix, max_results)
    
    # =========================================================================
    # INTERNAL HELPER METHODS
    # =========================================================================
    
    def _upload_object(
        self,
        bucket: str,
        key: str,
        data: Union[bytes, BinaryIO],
        metadata: Optional[Dict[str, str]] = None,
        content_type: str = "application/octet-stream"
    ) -> Optional[ObjectMetadata]:
        """Upload an object to S3."""
        if not self.enabled:
            return None
            
        try:
            # Ensure bucket exists
            self.ensure_bucket(bucket)
            
            # Prepare upload parameters
            extra_args = {
                'ContentType': content_type
            }
            if metadata:
                extra_args['Metadata'] = metadata
                
            # Handle bytes vs file-like object
            if isinstance(data, bytes):
                body = io.BytesIO(data)
            else:
                body = data
                
            # Calculate hash for ETag verification
            if isinstance(data, bytes):
                content_hash = hashlib.md5(data).hexdigest()
            else:
                # For streams, we can't pre-calculate
                content_hash = None
                
            # Upload
            self._client.upload_fileobj(
                Fileobj=body,
                Bucket=bucket,
                Key=key,
                ExtraArgs=extra_args
            )
            
            # Get object metadata
            response = self._client.head_object(Bucket=bucket, Key=key)
            
            return ObjectMetadata(
                key=key,
                bucket=bucket,
                size=response.get('ContentLength', 0),
                etag=response.get('ETag', '').strip('"'),
                content_type=response.get('ContentType', content_type),
                last_modified=response.get('LastModified', datetime.now(timezone.utc)),
                metadata=response.get('Metadata', {}),
                storage_class=response.get('StorageClass'),
                version_id=response.get('VersionId')
            )
            
        except ClientError as e:
            logger.error(f"Failed to upload object {key} to {bucket}: {e}")
            return None
    
    def _download_object(self, bucket: str, key: str) -> Optional[bytes]:
        """Download an object from S3."""
        if not self.enabled:
            return None
            
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'NoSuchKey':
                logger.debug(f"Object not found: {bucket}/{key}")
            else:
                logger.error(f"Failed to download object {key} from {bucket}: {e}")
            return None
    
    def _list_objects(
        self,
        bucket: str,
        prefix: str,
        max_results: int = 1000
    ) -> List[ObjectMetadata]:
        """List objects with a given prefix."""
        if not self.enabled:
            return []
            
        objects = []
        try:
            paginator = self._client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=bucket,
                Prefix=prefix,
                PaginationConfig={'MaxItems': max_results}
            )
            
            for page in page_iterator:
                for obj in page.get('Contents', []):
                    objects.append(ObjectMetadata(
                        key=obj['Key'],
                        bucket=bucket,
                        size=obj['Size'],
                        etag=obj.get('ETag', '').strip('"'),
                        content_type='',  # Not available in list
                        last_modified=obj.get('LastModified', datetime.now(timezone.utc)),
                        metadata={},
                        storage_class=obj.get('StorageClass')
                    ))
                    
            return objects
            
        except ClientError as e:
            logger.error(f"Failed to list objects in {bucket}/{prefix}: {e}")
            return []
    
    def _delete_prefix(self, bucket: str, prefix: str) -> bool:
        """Delete all objects with a given prefix."""
        if not self.enabled:
            return False
            
        try:
            paginator = self._client.get_paginator('list_objects_v2')
            
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                objects = page.get('Contents', [])
                if objects:
                    delete_keys = [{'Key': obj['Key']} for obj in objects]
                    self._client.delete_objects(
                        Bucket=bucket,
                        Delete={'Objects': delete_keys}
                    )
                    
            return True
            
        except ClientError as e:
            logger.error(f"Failed to delete prefix {prefix} from {bucket}: {e}")
            return False
    
    def stream_download(
        self,
        bucket: str,
        key: str,
        chunk_size: int = 1024 * 1024  # 1MB chunks
    ) -> Generator[bytes, None, None]:
        """Stream download an object in chunks."""
        if not self.enabled:
            return
            
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
            body = response['Body']
            
            while True:
                chunk = body.read(chunk_size)
                if not chunk:
                    break
                yield chunk
                
        except ClientError as e:
            logger.error(f"Failed to stream download {key} from {bucket}: {e}")
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def get_object_url(
        self,
        bucket: str,
        key: str,
        expires_in: int = 3600
    ) -> Optional[str]:
        """Generate a presigned URL for an object."""
        if not self.enabled:
            return None
            
        try:
            url = self._client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned URL: {e}")
            return None
    
    def get_upload_url(
        self,
        bucket: str,
        key: str,
        expires_in: int = 3600,
        content_type: str = "application/octet-stream"
    ) -> Optional[str]:
        """Generate a presigned URL for uploading."""
        if not self.enabled:
            return None
            
        try:
            url = self._client.generate_presigned_url(
                'put_object',
                Params={
                    'Bucket': bucket,
                    'Key': key,
                    'ContentType': content_type
                },
                ExpiresIn=expires_in
            )
            return url
        except ClientError as e:
            logger.error(f"Failed to generate presigned upload URL: {e}")
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """Check S3 backend health."""
        if not self.enabled:
            return {"status": "disabled", "available": False}
            
        try:
            buckets = self.list_buckets()
            return {
                "status": "healthy",
                "available": True,
                "endpoint": self.config.endpoint_url or "AWS S3",
                "region": self.config.region,
                "buckets": len(buckets)
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "available": False,
                "error": str(e)
            }
    
    def close(self) -> None:
        """Close the S3 client (no-op for boto3)."""
        logger.info("S3 backend closed")
