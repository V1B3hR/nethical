#!/usr/bin/env python3
"""Integration test for the end-to-end training pipeline.

This test creates sample datasets, processes them, and trains a model.
"""
import json
import shutil
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.dataset_processors.cyber_security_processor import (
    CyberSecurityAttacksProcessor,
)
from scripts.dataset_processors.microsoft_security_processor import (
    MicrosoftSecurityProcessor,
)
from nethical.mlops.baseline import BaselineMLClassifier


def test_end_to_end_pipeline():
    """Test the complete pipeline from raw CSV to trained model."""
    print("\n" + "=" * 70)
    print("  Integration Test: End-to-End Training Pipeline")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "data"
        data_dir.mkdir()

        # Step 1: Create sample datasets
        print("\n[1/5] Creating sample datasets...")

        # Cyber security attacks dataset
        attacks_csv = data_dir / "attacks.csv"
        with open(attacks_csv, "w") as f:
            f.write("attack_type,severity,anomaly_score,packet_count,label\n")
            for i in range(50):
                if i % 3 == 0:
                    f.write(f"Attack{i},High,{80+i%20},1000,malicious\n")
                else:
                    f.write(f"Normal{i},Low,{5+i%15},50,normal\n")

        # Microsoft incidents dataset
        incidents_csv = data_dir / "incidents.csv"
        with open(incidents_csv, "w") as f:
            f.write("IncidentGrade,Severity,AlertCount,DeviceId\n")
            for i in range(30):
                if i % 2 == 0:
                    f.write(f"TruePositive,High,{5+i%10},device_{i}\n")
                else:
                    f.write(f"FalsePositive,Low,{1+i%5},device_{i}\n")

        print(f"  ✓ Created {attacks_csv.name} (50 records)")
        print(f"  ✓ Created {incidents_csv.name} (30 records)")

        # Step 2: Process datasets
        print("\n[2/5] Processing datasets...")

        processed_dir = tmpdir / "processed"

        # Process attacks
        processor1 = CyberSecurityAttacksProcessor(output_dir=processed_dir)
        records1 = processor1.process(attacks_csv)
        file1 = processor1.save_processed_data(records1)
        print(f"  ✓ Processed attacks: {len(records1)} records")

        # Process incidents
        processor2 = MicrosoftSecurityProcessor(output_dir=processed_dir)
        records2 = processor2.process(incidents_csv)
        file2 = processor2.save_processed_data(records2)
        print(f"  ✓ Processed incidents: {len(records2)} records")

        # Step 3: Merge datasets
        print("\n[3/5] Merging datasets...")

        all_records = records1 + records2
        merged_file = tmpdir / "merged_data.json"

        with open(merged_file, "w") as f:
            json.dump(all_records, f, indent=2)

        print(f"  ✓ Merged {len(all_records)} total records")

        # Step 4: Train model
        print("\n[4/5] Training BaselineMLClassifier...")

        # Split data
        split = int(0.8 * len(all_records))
        train_data = all_records[:split]
        val_data = all_records[split:]

        # Train
        clf = BaselineMLClassifier()
        clf.train(train_data)

        print(f"  ✓ Trained on {len(train_data)} samples")
        print(f"  ✓ Feature weights learned:")
        for feature, weight in clf.feature_weights.items():
            print(f"      {feature:20s}: {weight:.4f}")

        # Step 5: Evaluate model
        print("\n[5/5] Evaluating model...")

        val_features = [d["features"] for d in val_data]
        val_labels = [d["label"] for d in val_data]
        val_preds = [clf.predict(f)["label"] for f in val_features]

        metrics = clf.compute_metrics(val_preds, val_labels)

        print(f"  Validation Metrics:")
        print(f"    Accuracy:  {metrics['accuracy']:.3f}")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall:    {metrics['recall']:.3f}")
        print(f"    F1 Score:  {metrics['f1_score']:.3f}")

        # Verify model works
        assert clf.trained is True
        assert clf.training_samples == len(train_data)
        assert (
            metrics["accuracy"] >= 0.5
        ), "Model should achieve better than random accuracy"

        # Step 6: Save and load model
        print("\n[6/6] Testing model save/load...")

        model_file = tmpdir / "model.json"
        clf.save(str(model_file))

        loaded_clf = BaselineMLClassifier.load(str(model_file))
        assert loaded_clf.trained is True
        assert loaded_clf.training_samples == clf.training_samples

        # Verify predictions match
        sample_features = val_features[0]
        pred1 = clf.predict(sample_features)
        pred2 = loaded_clf.predict(sample_features)
        assert pred1["label"] == pred2["label"]
        assert abs(pred1["score"] - pred2["score"]) < 0.001

        print(f"  ✓ Model saved and loaded successfully")

        print("\n" + "=" * 70)
        print("  ✓ All integration tests passed!")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    test_end_to_end_pipeline()
