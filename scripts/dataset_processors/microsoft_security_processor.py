"""Processor for Microsoft Security Incident Prediction dataset."""
import logging
from pathlib import Path
from typing import Dict, List, Any
from .base_processor import BaseDatasetProcessor

logger = logging.getLogger(__name__)


class MicrosoftSecurityProcessor(BaseDatasetProcessor):
    """Process the Microsoft Security Incident Prediction dataset.
    
    Dataset: https://www.kaggle.com/datasets/Microsoft/microsoft-security-incident-prediction
    """
    
    def __init__(self, output_dir: Path = Path("data/processed")):
        super().__init__("microsoft_security", output_dir)
    
    def process(self, input_path: Path) -> List[Dict[str, Any]]:
        """Process the dataset.
        
        Args:
            input_path: Path to CSV file
            
        Returns:
            List of processed records
        """
        logger.info(f"Processing Microsoft Security dataset from {input_path}")
        
        rows = self.load_csv(input_path)
        if not rows:
            logger.warning(f"No data found in {input_path}")
            return []
        
        records = []
        for i, row in enumerate(rows):
            try:
                features = self.extract_standard_features(row)
                label = self.extract_label(row)
                
                records.append({
                    'dataset': self.dataset_name,
                    'record_id': i,
                    'features': features,
                    'label': label
                })
            except Exception as e:
                logger.warning(f"Error processing row {i}: {e}")
                continue
        
        logger.info(f"Processed {len(records)} records from Microsoft Security")
        return records
    
    def extract_standard_features(self, row: Dict[str, Any]) -> Dict[str, float]:
        """Map Microsoft Security dataset fields to standard features."""
        
        # Microsoft datasets often have specific fields
        violation_count = 0.0
        if 'IncidentGrade' in row or 'AlertTitle' in row:
            violation_count = 0.5
        
        # Severity mapping
        severity_max = 0.0
        if 'IncidentGrade' in row:
            grade = str(row['IncidentGrade']).lower()
            if grade in ['truepositive', 'true positive', 'tp']:
                severity_max = 0.8
            elif grade in ['benignpositive', 'benign positive', 'bp']:
                severity_max = 0.3
            elif grade in ['falsepositive', 'false positive', 'fp']:
                severity_max = 0.1
        
        if 'Severity' in row:
            sev = str(row['Severity']).lower()
            if 'high' in sev:
                severity_max = max(severity_max, 0.9)
            elif 'medium' in sev:
                severity_max = max(severity_max, 0.6)
            elif 'low' in sev:
                severity_max = max(severity_max, 0.3)
        
        # Entity features
        recency_score = 0.5
        
        # Alert counts
        frequency_score = 0.0
        if 'AlertCount' in row:
            frequency_score = self.normalize_feature(row.get('AlertCount', 0), 0, 100)
        
        # Context from device or user information
        context_risk = 0.0
        if any(key in row for key in ['DeviceId', 'OrgId', 'DetectorId']):
            context_risk = 0.4
        
        return {
            'violation_count': violation_count,
            'severity_max': severity_max,
            'recency_score': recency_score,
            'frequency_score': frequency_score,
            'context_risk': context_risk
        }
    
    def extract_label(self, row: Dict[str, Any]) -> int:
        """Extract label from Microsoft dataset.
        
        Returns:
            1 for true positive incidents, 0 for benign/false positives
        """
        # Check for incident grade
        if 'IncidentGrade' in row:
            grade = str(row['IncidentGrade']).lower()
            if 'true' in grade and 'positive' in grade:
                return 1
            else:
                return 0
        
        # Check for other label indicators
        if 'HasIncident' in row:
            return 1 if str(row['HasIncident']).lower() in ['1', 'true', 'yes'] else 0
        
        # Default to safe
        return 0
