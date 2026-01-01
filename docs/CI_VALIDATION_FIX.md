# CI Validation AttributeError Fix

## Problem

The CI validation was failing with `AttributeError: MODIFY_CODE` when importing from `nethical.core.semantic_primitives`. However, the source file correctly defines `MODIFY_CODE` in the `SemanticPrimitive` enum.

This indicated a mismatch between the file version seen by the test runner and the latest source code, likely caused by:
- Stale Python bytecode (`.pyc` files)
- Cached `__pycache__` directories
- Module shadowing or outdated installations
- Pip cache issues during package installation

## Solution

The fix implements the following steps in both CI and validation workflows:

### 1. Added Security Permissions
Added explicit permissions blocks at both workflow and job levels with minimum required permissions (`contents: read`) to follow security best practices.

### 2. Clear Python Cache and Bytecode
Before running tests, all `.pyc` files and `__pycache__` directories are removed:
```bash
find . -type f -name '*.pyc' -delete
find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
```

### 3. Verify Source File Content
Added a verification step to print the `SemanticPrimitive` enum definition from the source file:
```bash
cat nethical/core/semantic_primitives.py | grep -A 25 "class SemanticPrimitive"
grep -n "MODIFY_CODE" nethical/core/semantic_primitives.py
```

This ensures the test runner sees the correct source file content.

### 4. Force-Reinstall Package
After installing dependencies, force-reinstall the nethical package without dependencies:
```bash
pip install . --force-reinstall --no-deps
```

This ensures the test runner uses the current checked-out commit, not a cached installation.

### 5. Verify Python Environment
Added extensive environment verification to aid debugging:
- Python version and executable path
- Installed nethical package location
- sys.path contents
- SemanticPrimitive enum members
- semantic_primitives.py module location

Example output:
```python
python -c "from nethical.core.semantic_primitives import SemanticPrimitive; 
          print('Members:', list(SemanticPrimitive.__members__.keys())); 
          print('Has MODIFY_CODE:', 'MODIFY_CODE' in SemanticPrimitive.__members__)"
```

## Files Modified

### `.github/workflows/ci.yml`
- Added workflow-level and job-level permissions
- Added cache clearing step before tests
- Added source file verification step
- Added force-reinstall step after dependency installation
- Added Python environment verification step

### `.github/workflows/validation.yml`
- Applied the same changes as ci.yml to maintain consistency

### `tests/unit/test_semantic_primitives_import.py` (New)
Created comprehensive tests to verify:
- `MODIFY_CODE` attribute exists in `SemanticPrimitive` enum
- `MODIFY_CODE` has the correct value
- All expected enum members are present
- Enum can be accessed by name and value
- `EnhancedPrimitiveDetector` correctly identifies `MODIFY_CODE` primitive

## Benefits

1. **Prevents Stale Cache Issues**: Clears Python bytecode before every test run
2. **Ensures Correct Version**: Force-reinstall guarantees tests run against checked-out commit
3. **Easier Debugging**: Extensive verification output helps identify issues quickly
4. **Better Security**: Explicit permissions follow GitHub Actions security best practices
5. **Comprehensive Testing**: New tests specifically verify `MODIFY_CODE` attribute accessibility

## Testing

The solution was tested locally:
1. All cache clearing commands work without errors
2. Source verification correctly displays enum definition
3. Force-reinstall completes successfully
4. Environment verification outputs expected information
5. All 10 new tests pass successfully

## Future Considerations

If similar issues occur with other modules:
1. Apply the same cache clearing pattern to other workflows
2. Consider adding a reusable workflow template for common setup steps
3. Monitor for any performance impact from force-reinstalling (currently minimal)

## References

- GitHub Issue: CI validation failing with AttributeError: MODIFY_CODE
- Source file: `nethical/core/semantic_primitives.py` (line 27)
- Test file: `tests/unit/test_semantic_primitives_import.py`
