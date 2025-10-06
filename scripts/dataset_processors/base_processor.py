"""Base dataset processor with common utilities."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import csv

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class BaseDatasetProcessor:
    """Base class for dataset processors."""
    
    def __init__(self, dataset_name: str, output_dir: Path = Path("data/processed")):
        """Initialize processor.
        
        Args:
            dataset_name: Name of the dataset
            output_dir: Directory to save processed data
        """
        self.dataset_name = dataset_name
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def process(self, input_path: Path) -> List[Dict[str, Any]]:
        """Process dataset and return standardized records.
        
        Args:
            input_path: Path to the raw dataset file
            
        Returns:
            List of processed records with 'features' and 'label' keys
        """
        raise NotImplementedError("Subclasses must implement process()")
    
    def save_processed_data(self, records: List[Dict[str, Any]]) -> Path:
        """Save processed records to JSON.
        
        Args:
            records: List of processed records
            
        Returns:
            Path to saved file
        """
        output_file = self.output_dir / f"{self.dataset_name}_processed.json"
        with open(output_file, 'w') as f:
            json.dump(records, f, indent=2)
        logger.info(f"Saved {len(records)} records to {output_file}")
        return output_file
    
    def load_csv(self, path: Path, encoding: str = 'utf-8') -> List[Dict[str, Any]]:
        """Load CSV file into list of dictionaries.
        
        Args:
            path: Path to CSV file
            encoding: File encoding
            
        Returns:
            List of row dictionaries
        """
        rows = []
        try:
            with open(path, 'r', encoding=encoding) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except UnicodeDecodeError:
            # Try alternative encodings
            for enc in ['latin-1', 'iso-8859-1', 'cp1252']:
                try:
                    with open(path, 'r', encoding=enc) as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                    logger.info(f"Successfully read {path} with encoding {enc}")
                    break
                except UnicodeDecodeError:
                    continue
        return rows
    
    def normalize_feature(self, value: Any, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Normalize a feature value to [0, 1] range.
        
        Args:
            value: Feature value
            min_val: Minimum expected value
            max_val: Maximum expected value
            
        Returns:
            Normalized value
        """
        try:
            val = float(value)
            if max_val == min_val:
                return 0.5
            normalized = (val - min_val) / (max_val - min_val)
            return max(0.0, min(1.0, normalized))
        except (ValueError, TypeError):
            return 0.0
    
    def extract_standard_features(self, row: Dict[str, Any]) -> Dict[str, float]:
        """Extract standard features from a row.
        
        This should be overridden by subclasses to map dataset-specific
        fields to our standard feature set:
        - violation_count: Number/frequency of violations
        - severity_max: Maximum severity level
        - recency_score: How recent the event is
        - frequency_score: Frequency of similar events
        - context_risk: Contextual risk factors
        
        Args:
            row: Raw data row
            
        Returns:
            Dictionary of standard features
        """
        return {
            'violation_count': 0.0,
            'severity_max': 0.0,
            'recency_score': 0.0,
            'frequency_score': 0.0,
            'context_risk': 0.0
        }
    
    def extract_label(self, row: Dict[str, Any]) -> int:
        """Extract binary label from a row.
        
        This should be overridden by subclasses to determine if the
        event should be labeled as risky (1) or safe (0).
        
        Args:
            row: Raw data row
            
        Returns:
            Binary label (0 or 1)
        """
        return 0
