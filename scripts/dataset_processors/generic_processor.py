"""Generic processor for security datasets."""
import logging
from pathlib import Path
from typing import Dict, List, Any
from .base_processor import BaseDatasetProcessor

logger = logging.getLogger(__name__)


class GenericSecurityProcessor(BaseDatasetProcessor):
    """Generic processor for security datasets with flexible field mapping."""
    
    def __init__(self, dataset_name: str = "generic_security", output_dir: Path = Path("data/processed")):
        super().__init__(dataset_name, output_dir)
    
    def process(self, input_path: Path) -> List[Dict[str, Any]]:
        """Process the dataset.
        
        Args:
            input_path: Path to CSV file
            
        Returns:
            List of processed records
        """
        logger.info(f"Processing {self.dataset_name} dataset from {input_path}")
        
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
        
        logger.info(f"Processed {len(records)} records from {self.dataset_name}")
        return records
    
    def extract_standard_features(self, row: Dict[str, Any]) -> Dict[str, float]:
        """Map generic security dataset fields to standard features.
        
        Uses heuristics to identify relevant fields.
        """
        # Initialize features
        violation_count = 0.0
        severity_max = 0.0
        recency_score = 0.5
        frequency_score = 0.0
        context_risk = 0.0
        
        # Scan row for relevant indicators
        row_lower = {k.lower(): v for k, v in row.items()}
        
        # Look for severity indicators
        severity_keys = ['severity', 'priority', 'risk', 'level', 'grade', 'impact']
        for key in severity_keys:
            if key in row_lower:
                val = str(row_lower[key]).lower()
                if any(x in val for x in ['critical', 'high', 'severe', '4', '5']):
                    severity_max = 0.9
                elif any(x in val for x in ['medium', 'moderate', '3']):
                    severity_max = 0.6
                elif any(x in val for x in ['low', 'minor', '1', '2']):
                    severity_max = 0.3
                break
        
        # Look for violation/attack indicators
        violation_keys = ['attack', 'threat', 'alert', 'violation', 'incident', 'anomaly']
        for key in violation_keys:
            if key in row_lower:
                violation_count = 0.6
                break
        
        # Look for frequency indicators
        freq_keys = ['count', 'frequency', 'number', 'total']
        for key in freq_keys:
            if key in row_lower:
                try:
                    val = float(row_lower[key])
                    frequency_score = self.normalize_feature(val, 0, 100)
                    break
                except (ValueError, TypeError):
                    pass
        
        # Look for context indicators
        context_keys = ['source', 'destination', 'user', 'device', 'host', 'ip', 'port', 'protocol']
        if any(key in row_lower for key in context_keys):
            context_risk = 0.4
        
        return {
            'violation_count': violation_count,
            'severity_max': severity_max,
            'recency_score': recency_score,
            'frequency_score': frequency_score,
            'context_risk': context_risk
        }
    
    def extract_label(self, row: Dict[str, Any]) -> int:
        """Extract label using heuristics.
        
        Returns:
            1 for malicious/risky, 0 for normal/benign
        """
        row_lower = {k.lower(): v for k, v in row.items()}
        
        # Check for explicit label fields
        label_keys = ['label', 'class', 'target', 'prediction', 'result', 'classification']
        for key in label_keys:
            if key in row_lower:
                val = str(row_lower[key]).lower()
                
                # Malicious indicators
                malicious_indicators = [
                    'malicious', 'attack', 'threat', 'anomaly', 'intrusion',
                    'breach', 'suspicious', 'true', 'positive', '1', 'yes',
                    'bad', 'harmful', 'dangerous'
                ]
                if any(indicator in val for indicator in malicious_indicators):
                    return 1
                
                # Benign indicators
                benign_indicators = [
                    'benign', 'normal', 'safe', 'legitimate', 'false', 'negative',
                    '0', 'no', 'good', 'clean'
                ]
                if any(indicator in val for indicator in benign_indicators):
                    return 0
        
        # If no clear label, check for severity-based classification
        if 'severity' in row_lower or 'priority' in row_lower:
            key = 'severity' if 'severity' in row_lower else 'priority'
            val = str(row_lower[key]).lower()
            if any(x in val for x in ['high', 'critical', 'severe']):
                return 1
        
        # Conservative default: assume benign if unclear
        return 0
