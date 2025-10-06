# Train Model Real Data Implementation Summary

## Overview
Modified `scripts/train_model.py` to use real-world datasets by default instead of synthetic data, specifically targeting the two datasets requested:
- https://www.kaggle.com/code/kmldas/data-ethics-in-data-science-analytics-ml-and-ai
- https://www.kaggle.com/datasets/xontoloyo/security-breachhh

## Changes Made

### 1. Updated `scripts/train_model.py`

#### Added Functionality
- **`load_real_world_data()` function**: New function that:
  - Attempts to download datasets using Kaggle API (if available)
  - Processes all CSV files found in `data/external/` directory
  - Uses `GenericSecurityProcessor` to handle both datasets
  - Falls back to synthetic data if no real datasets are found
  - Returns processed records with standardized features and labels

- **Import additions**: Added `GenericSecurityProcessor` import to enable dataset processing

- **Modified `main()` function**: 
  - Replaced synthetic data generation with `load_real_world_data()` call
  - Added informative messages about which datasets are being loaded
  - Maintains all existing training, evaluation, and promotion gate logic

#### Updated Documentation
- Rewrote docstring to emphasize real-world data usage
- Explained the automatic download and fallback mechanism
- Provided manual setup instructions for when Kaggle API is unavailable

### 2. Updated Documentation Files

#### `docs/TRAINING_GUIDE.md`
- Added Option 1 section for `train_model.py` with real datasets
- Explained the two specific datasets being used
- Provided manual download instructions
- Maintained existing `baseline_orchestrator.py` documentation

#### `scripts/README.md`
- Updated `train_model.py` section to reflect real-world data usage
- Added manual dataset setup instructions
- Clarified the difference between `train_model.py` (specific datasets) and `baseline_orchestrator.py` (all datasets)

### 3. Added Test Suite

#### `tests/test_train_model_real_data.py`
- Created comprehensive test for real data loading functionality
- Tests data processing, structure validation, and training
- Tests fallback to synthetic data when no real datasets available
- All tests pass successfully

## How It Works

### Automatic Workflow
1. User runs `python scripts/train_model.py`
2. Script attempts to download the two datasets using Kaggle API
3. If Kaggle API unavailable, script looks for CSV files in `data/external/`
4. Processes all found CSV files using `GenericSecurityProcessor`
5. Trains model on all available real data
6. Falls back to synthetic data if no real datasets found

### Manual Workflow (No Kaggle API)
1. User manually downloads datasets from Kaggle
2. Extracts CSV files to `data/external/` directory
3. Runs `python scripts/train_model.py`
4. Script processes the CSV files and trains the model

## Data Processing

The `GenericSecurityProcessor` extracts standard features from any CSV file:
- **violation_count**: Detected from attack/threat/alert indicators
- **severity_max**: Mapped from severity/priority/risk fields
- **recency_score**: Default value (0.5) for generic datasets
- **frequency_score**: Extracted from count/frequency fields
- **context_risk**: Presence of source/destination/user fields

Labels are extracted using heuristics:
- Malicious (1): malicious, attack, threat, high severity, etc.
- Benign (0): benign, normal, safe, low severity, etc.

## Testing

All functionality verified through:
1. ✅ Manual testing with sample CSV files
2. ✅ Automated test suite (`test_train_model_real_data.py`)
3. ✅ Syntax validation
4. ✅ Full training pipeline execution

Sample test output shows:
- Successfully loads and processes CSV files
- Generates correct feature structure
- Trains model successfully
- Falls back to synthetic data when needed

## Backward Compatibility

The changes maintain full backward compatibility:
- `generate_synthetic_labeled_data()` function still available
- All existing tests still pass
- Fallback mechanism ensures training always works
- Same command-line interface and options

## Usage Examples

### Basic Training
```bash
python scripts/train_model.py
```

### Full Workflow (Training + Testing)
```bash
python scripts/train_model.py --run-all
```

### With Manual Dataset Setup
```bash
# Download datasets manually and place in data/external/
python scripts/train_model.py
```

## Files Modified
- `scripts/train_model.py` - Added real data loading functionality
- `docs/TRAINING_GUIDE.md` - Updated documentation
- `scripts/README.md` - Updated usage instructions

## Files Created
- `tests/test_train_model_real_data.py` - Comprehensive test suite

## Minimal Impact
The implementation follows the principle of minimal changes:
- Only ~120 lines added to `train_model.py`
- Reuses existing `GenericSecurityProcessor` infrastructure
- No changes to core training/evaluation logic
- No breaking changes to existing functionality
