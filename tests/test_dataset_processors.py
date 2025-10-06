"""Test dataset processors."""
import json
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.dataset_processors.cyber_security_processor import CyberSecurityAttacksProcessor
from scripts.dataset_processors.microsoft_security_processor import MicrosoftSecurityProcessor
from scripts.dataset_processors.generic_processor import GenericSecurityProcessor


def test_cyber_security_processor():
    """Test cyber security dataset processor."""
    print("\n=== Testing CyberSecurityAttacksProcessor ===")
    
    # Create a sample CSV file
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test_attacks.csv"
        with open(csv_file, 'w') as f:
            f.write("attack_type,severity,anomaly_score,packet_count,label\n")
            f.write("DDoS,High,85,1500,malicious\n")
            f.write("Normal,Low,10,50,normal\n")
            f.write("Port Scan,Medium,60,200,attack\n")
        
        processor = CyberSecurityAttacksProcessor(output_dir=Path(tmpdir))
        records = processor.process(csv_file)
        
        print(f"Processed {len(records)} records")
        assert len(records) == 3, f"Expected 3 records, got {len(records)}"
        
        # Check first record (malicious)
        rec1 = records[0]
        print(f"\nRecord 1: {json.dumps(rec1, indent=2)}")
        assert rec1['label'] == 1, "First record should be labeled as malicious"
        assert rec1['features']['severity_max'] > 0.5, "High severity should map to >0.5"
        
        # Check second record (normal)
        rec2 = records[1]
        print(f"\nRecord 2: {json.dumps(rec2, indent=2)}")
        assert rec2['label'] == 0, "Second record should be labeled as normal"
        
        print("✓ CyberSecurityAttacksProcessor test passed")


def test_microsoft_processor():
    """Test Microsoft security dataset processor."""
    print("\n=== Testing MicrosoftSecurityProcessor ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test_incidents.csv"
        with open(csv_file, 'w') as f:
            f.write("IncidentGrade,Severity,AlertCount,DeviceId\n")
            f.write("TruePositive,High,5,device1\n")
            f.write("FalsePositive,Low,1,device2\n")
            f.write("BenignPositive,Medium,2,device3\n")
        
        processor = MicrosoftSecurityProcessor(output_dir=Path(tmpdir))
        records = processor.process(csv_file)
        
        print(f"Processed {len(records)} records")
        assert len(records) == 3, f"Expected 3 records, got {len(records)}"
        
        # Check labels
        print(f"\nRecord 1 (TruePositive): label={records[0]['label']}")
        assert records[0]['label'] == 1, "TruePositive should be labeled as 1"
        
        print(f"Record 2 (FalsePositive): label={records[1]['label']}")
        assert records[1]['label'] == 0, "FalsePositive should be labeled as 0"
        
        print("✓ MicrosoftSecurityProcessor test passed")


def test_generic_processor():
    """Test generic security dataset processor."""
    print("\n=== Testing GenericSecurityProcessor ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test_generic.csv"
        with open(csv_file, 'w') as f:
            f.write("severity,label,count,source\n")
            f.write("critical,malicious,100,10.0.0.1\n")
            f.write("low,benign,5,192.168.1.1\n")
            f.write("high,threat,50,172.16.0.1\n")
        
        processor = GenericSecurityProcessor("test_generic", output_dir=Path(tmpdir))
        records = processor.process(csv_file)
        
        print(f"Processed {len(records)} records")
        assert len(records) == 3, f"Expected 3 records, got {len(records)}"
        
        # Check severity mapping
        print(f"\nRecord 1 (critical): severity_max={records[0]['features']['severity_max']}")
        assert records[0]['features']['severity_max'] >= 0.8, "Critical severity should be high"
        
        # Check label extraction
        print(f"Record 1 label: {records[0]['label']}")
        assert records[0]['label'] == 1, "Malicious should be labeled as 1"
        
        print(f"Record 2 label: {records[1]['label']}")
        assert records[1]['label'] == 0, "Benign should be labeled as 0"
        
        print("✓ GenericSecurityProcessor test passed")


def test_feature_extraction():
    """Test feature extraction produces valid values."""
    print("\n=== Testing Feature Extraction ===")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = Path(tmpdir) / "test.csv"
        with open(csv_file, 'w') as f:
            f.write("severity,label\n")
            f.write("high,malicious\n")
        
        processor = GenericSecurityProcessor("test", output_dir=Path(tmpdir))
        records = processor.process(csv_file)
        
        features = records[0]['features']
        print(f"Features: {json.dumps(features, indent=2)}")
        
        # All features should be in [0, 1] range
        for key, value in features.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} is out of range [0, 1]"
        
        # Should have all standard features
        expected_keys = {'violation_count', 'severity_max', 'recency_score', 'frequency_score', 'context_risk'}
        assert set(features.keys()) == expected_keys, f"Missing features: {expected_keys - set(features.keys())}"
        
        print("✓ Feature extraction test passed")


if __name__ == "__main__":
    print("Testing Dataset Processors")
    print("=" * 60)
    
    test_cyber_security_processor()
    test_microsoft_processor()
    test_generic_processor()
    test_feature_extraction()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
