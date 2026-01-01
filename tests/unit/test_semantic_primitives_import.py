"""Tests for semantic primitives import and enum members."""

from nethical.core.semantic_primitives import SemanticPrimitive, EnhancedPrimitiveDetector


class TestSemanticPrimitiveEnum:
    """Test SemanticPrimitive enum and its members."""
    
    def test_modify_code_exists(self):
        """Test that MODIFY_CODE attribute exists in SemanticPrimitive enum."""
        assert hasattr(SemanticPrimitive, 'MODIFY_CODE'), "MODIFY_CODE should exist in SemanticPrimitive"
        assert 'MODIFY_CODE' in SemanticPrimitive.__members__, "MODIFY_CODE should be in enum members"
    
    def test_modify_code_value(self):
        """Test that MODIFY_CODE has the correct value."""
        assert SemanticPrimitive.MODIFY_CODE.value == "MODIFY_CODE"
    
    def test_critical_members_exist(self):
        """Test that critical members including MODIFY_CODE exist in the enum."""
        # Test only critical members to make test more resilient to enum additions
        critical_members = [
            'MODIFY_CODE',  # The key member we're testing
            'ACCESS_USER_DATA',
            'EXECUTE_CODE',
            'GENERATE_CODE',
            'ACCESS_SYSTEM',
        ]
        
        actual_members = list(SemanticPrimitive.__members__.keys())
        
        for member in critical_members:
            assert member in actual_members, f"{member} should be in SemanticPrimitive members"
    
    def test_enum_has_minimum_members(self):
        """Test that enum has at least the expected minimum number of members."""
        # Ensure we have at least the core primitives (allows for future additions)
        min_expected_members = 20
        actual_count = len(SemanticPrimitive)
        assert actual_count >= min_expected_members, (
            f"Expected at least {min_expected_members} members, got {actual_count}"
        )
    
    def test_enum_access_by_name(self):
        """Test that we can access MODIFY_CODE by name."""
        member = SemanticPrimitive['MODIFY_CODE']
        assert member == SemanticPrimitive.MODIFY_CODE
    
    def test_enum_access_by_value(self):
        """Test that we can access MODIFY_CODE by value."""
        member = SemanticPrimitive("MODIFY_CODE")
        assert member == SemanticPrimitive.MODIFY_CODE


class TestEnhancedPrimitiveDetector:
    """Test EnhancedPrimitiveDetector with MODIFY_CODE."""
    
    def test_detector_initialization(self):
        """Test that detector can be initialized."""
        detector = EnhancedPrimitiveDetector(use_embedding_similarity=False)
        assert detector is not None
    
    def test_detect_modify_code_primitive(self):
        """Test that detector can identify MODIFY_CODE primitive."""
        detector = EnhancedPrimitiveDetector(use_embedding_similarity=False)
        
        # Test with text that should trigger MODIFY_CODE detection
        action_text = "modify the code in the function"
        primitives = detector.detect_primitives(action_text)
        
        # Should detect MODIFY_CODE primitive
        assert SemanticPrimitive.MODIFY_CODE in primitives, (
            f"MODIFY_CODE should be detected in '{action_text}', got {primitives}"
        )
    
    def test_detect_edit_code_primitive(self):
        """Test that detector identifies code editing as MODIFY_CODE."""
        detector = EnhancedPrimitiveDetector(use_embedding_similarity=False)
        
        action_text = "edit the implementation of the class"
        primitives = detector.detect_primitives(action_text)
        
        assert SemanticPrimitive.MODIFY_CODE in primitives
    
    def test_detect_refactor_code_primitive(self):
        """Test that detector identifies refactoring as MODIFY_CODE."""
        detector = EnhancedPrimitiveDetector(use_embedding_similarity=False)
        
        action_text = "refactor the code to improve performance"
        primitives = detector.detect_primitives(action_text)
        
        assert SemanticPrimitive.MODIFY_CODE in primitives
