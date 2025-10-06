"""Processor for Cyber Security Attacks dataset from Kaggle."""
import logging
from pathlib import Path
from typing import Dict, List, Any
from .base_processor import BaseDatasetProcessor

logger = logging.getLogger(__name__)


class CyberSecurityAttacksProcessor(BaseDatasetProcessor):
    """Process the Cyber Security Attacks dataset.
    
    Dataset: https://www.kaggle.com/datasets/teamincribo/cyber-security-attacks
    """
    
    def __init__(self, output_dir: Path = Path("data/processed")):
        super().__init__("cyber_security_attacks", output_dir)
    
    def process(self, input_path: Path) -> List[Dict[str, Any]]:
        """Process the dataset.
        
        Args:
            input_path: Path to CSV file
            
        Returns:
            List of processed records
        """
        logger.info(f"Processing Cyber Security Attacks dataset from {input_path}")
        
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
        
        logger.info(f"Processed {len(records)} records from Cyber Security Attacks")
        return records
    
    def extract_standard_features(self, row: Dict[str, Any]) -> Dict[str, float]:
        """Map dataset fields to standard features.
        
        Common fields in security attack datasets:
        - Attack type, severity, source IP, destination IP, protocol, etc.
        """
        # Map various possible field names to our standard features
        # These are common in cyber security datasets
        
        # Violation count based on attack indicators
        violation_count = 0.0
        if any(key in row for key in ['attack_type', 'Attack Type', 'type']):
            violation_count = 0.5
        if any(key in row for key in ['anomaly_score', 'Anomaly Scores', 'score']):
            score_key = next((k for k in ['anomaly_score', 'Anomaly Scores', 'score'] if k in row), None)
            if score_key:
                violation_count = self.normalize_feature(row.get(score_key, 0), 0, 100)
        
        # Severity based on severity/priority fields
        severity_max = 0.0
        severity_keys = ['severity', 'Severity Level', 'priority', 'Priority', 'level', 'Level']
        for key in severity_keys:
            if key in row:
                val = row[key]
                if isinstance(val, str):
                    val_lower = val.lower()
                    if 'critical' in val_lower or 'high' in val_lower:
                        severity_max = 0.9
                    elif 'medium' in val_lower:
                        severity_max = 0.6
                    elif 'low' in val_lower:
                        severity_max = 0.3
                else:
                    severity_max = self.normalize_feature(val, 0, 10)
                break
        
        # Recency score (if timestamp available)
        recency_score = 0.5  # Default middle value
        
        # Frequency based on packet count or similar
        frequency_score = 0.0
        freq_keys = ['packet_count', 'Packet Count', 'count', 'Count', 'frequency']
        for key in freq_keys:
            if key in row:
                frequency_score = self.normalize_feature(row.get(key, 0), 0, 1000)
                break
        
        # Context risk from protocol, port, or other indicators
        context_risk = 0.0
        if any(key in row for key in ['protocol', 'Protocol', 'port', 'Port']):
            context_risk = 0.4  # Moderate risk for network-based attacks
        
        return {
            'violation_count': violation_count,
            'severity_max': severity_max,
            'recency_score': recency_score,
            'frequency_score': frequency_score,
            'context_risk': context_risk
        }
    
    def extract_label(self, row: Dict[str, Any]) -> int:
        """Determine if this is a malicious/risky event.
        
        Args:
            row: Data row
            
        Returns:
            1 for attack/risky, 0 for normal
        """
        # Look for attack indicators
        attack_indicators = ['malicious', 'attack', 'intrusion', 'anomaly', 'threat']
        
        # Check common label fields
        label_keys = ['label', 'Label', 'class', 'Class', 'target', 'Target', 'attack_type', 'Attack Type']
        for key in label_keys:
            if key in row:
                val = str(row[key]).lower()
                # If any attack indicator is in the value, label as 1
                if any(indicator in val for indicator in attack_indicators):
                    return 1
                # Check for explicit normal/benign indicators
                if any(normal in val for normal in ['normal', 'benign', 'safe', 'legitimate']):
                    return 0
        
        # Default to risky if no clear indicator (conservative approach)
        return 1
