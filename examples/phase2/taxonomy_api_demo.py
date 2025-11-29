"""Example: Using the Taxonomy API for industry-specific ethical assessment.

This example demonstrates:
1. Creating industry-specific taxonomies
2. Validating taxonomy configurations
3. Tagging violations with ethical dimensions
4. Generating coverage reports
"""

from nethical.api.taxonomy_api import TaxonomyAPI
import json


def main():
    """Run taxonomy API examples."""
    
    print("=" * 80)
    print("Nethical Phase 2: Taxonomy API Examples")
    print("=" * 80)
    
    # Initialize API
    api = TaxonomyAPI("taxonomies/ethics_taxonomy.json")
    
    # Example 1: Validate current taxonomy
    print("\n1. Validating current taxonomy...")
    result = api.validate_taxonomy_endpoint()
    print(f"   Valid: {result['success']}")
    if result['success']:
        print(f"   Dimension count: {result['data']['stats']['dimension_count']}")
        print(f"   Mapping count: {result['data']['stats']['mapping_count']}")
    
    # Example 2: Get JSON Schema
    print("\n2. Retrieving JSON Schema...")
    schema_result = api.get_schema_endpoint()
    if schema_result['success']:
        print(f"   Schema ID: {schema_result['data']['schema']['$id']}")
        print(f"   Required fields: {', '.join(schema_result['data']['schema']['required'])}")
    
    # Example 3: Create healthcare taxonomy
    print("\n3. Creating healthcare-specific taxonomy...")
    healthcare_result = api.create_industry_taxonomy_endpoint("healthcare", "1.0")
    if healthcare_result['success']:
        taxonomy = healthcare_result['data']['taxonomy']
        print(f"   Industry: {taxonomy['industry']}")
        print(f"   Dimensions added:")
        for dim_name in ['patient_safety', 'medical_privacy']:
            if dim_name in taxonomy['dimensions']:
                print(f"     - {dim_name}: {taxonomy['dimensions'][dim_name]['description']}")
    
    # Example 4: Tag a violation
    print("\n4. Tagging a violation with ethical dimensions...")
    violation_type = "unauthorized_data_access"
    context = {
        "sensitive": True,
        "personal_data": True
    }
    tag_result = api.tag_violation_endpoint(violation_type, context)
    if tag_result['success']:
        print(f"   Violation: {violation_type}")
        print(f"   Primary dimension: {tag_result['data']['primary_dimension']}")
        print(f"   Dimension scores:")
        for tag in tag_result['data']['tags']:
            print(f"     - {tag['dimension']}: {tag['score']:.2f} (confidence: {tag['confidence']:.2f})")
    
    print("\n" + "=" * 80)
    print("Taxonomy API examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
