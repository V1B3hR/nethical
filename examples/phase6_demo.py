"""Phase 6 Demo: ML Assisted Enforcement

Demonstrates:
- Risk blending (0.7 * rules + 0.3 * ml)
- Gray zone detection and ML influence
- Pre/post decision audit trail
- FP delta tracking and gate checks
"""

import random
from nethical.core import MLBlendedRiskEngine, RiskZone, MLShadowClassifier


def print_section(title):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}\n")


def simulate_action_features(risk_level='medium'):
    """Simulate feature extraction for an action."""
    base_features = {
        'violation_count': 0.0,
        'severity_max': 0.0,
        'recency_score': 0.0,
        'frequency_score': 0.0,
        'context_risk': 0.0
    }
    
    if risk_level == 'low':
        base_features['violation_count'] = random.uniform(0.0, 0.3)
        base_features['severity_max'] = random.uniform(0.0, 0.2)
    elif risk_level == 'medium':
        base_features['violation_count'] = random.uniform(0.3, 0.6)
        base_features['severity_max'] = random.uniform(0.3, 0.6)
        base_features['recency_score'] = random.uniform(0.2, 0.5)
    else:  # high
        base_features['violation_count'] = random.uniform(0.6, 1.0)
        base_features['severity_max'] = random.uniform(0.7, 1.0)
        base_features['recency_score'] = random.uniform(0.5, 1.0)
    
    return base_features


def simulate_rule_score(features):
    """Simulate rule-based risk scoring."""
    return (
        features['violation_count'] * 0.4 +
        features['severity_max'] * 0.3 +
        features['recency_score'] * 0.2 +
        features['frequency_score'] * 0.1
    )


def simulate_ml_score(features):
    """Simulate ML risk scoring."""
    # ML might have different perspective
    return min(
        (features['violation_count'] * 0.3 +
         features['severity_max'] * 0.3 +
         features['recency_score'] * 0.25 +
         features['frequency_score'] * 0.15) * random.uniform(0.9, 1.1),
        1.0
    )


def score_to_classification(score):
    """Convert score to classification."""
    if score >= 0.7:
        return "deny"
    elif score >= 0.4:
        return "warn"
    else:
        return "allow"


def demo_blended_risk():
    """Demonstrate ML-assisted blended risk enforcement."""
    print_section("Phase 6: ML Assisted Enforcement Demo")
    
    # Initialize blended risk engine
    print("1. Initializing ML Blended Risk Engine...")
    blended_engine = MLBlendedRiskEngine(
        gray_zone_lower=0.4,
        gray_zone_upper=0.6,
        rule_weight=0.7,
        ml_weight=0.3,
        storage_path="./demo_data/phase6_blended",
        enable_ml_blending=True
    )
    print("✓ Blended risk engine initialized")
    print(f"  Gray zone: [{blended_engine.gray_zone_lower}, {blended_engine.gray_zone_upper}]")
    print(f"  Weights: {blended_engine.rule_weight} * rules + {blended_engine.ml_weight} * ML")
    
    # Demonstrate different risk zones
    print("\n2. Processing Actions Across Risk Zones...")
    
    zones_demo = [
        ('Clear Allow Zone', 0.25, 'low'),
        ('Gray Zone (ML assists)', 0.50, 'medium'),
        ('Clear Deny Zone', 0.75, 'high'),
    ]
    
    for zone_name, target_score, risk_level in zones_demo:
        print(f"\n   {zone_name}:")
        
        # Generate scenario
        agent_id = f"agent_{zone_name.replace(' ', '_')}"
        action_id = f"action_001"
        features = simulate_action_features(risk_level)
        
        # Adjust to get target score
        rule_score = target_score
        ml_score = target_score + random.uniform(-0.1, 0.1)
        ml_score = max(0.0, min(1.0, ml_score))
        
        rule_class = score_to_classification(rule_score)
        
        # Compute blended decision
        decision = blended_engine.compute_blended_risk(
            agent_id=agent_id,
            action_id=action_id,
            rule_risk_score=rule_score,
            rule_classification=rule_class,
            ml_risk_score=ml_score,
            ml_confidence=0.85,
            features=features
        )
        
        print(f"     Rule Score: {decision.rule_risk_score:.3f} → {decision.rule_classification}")
        print(f"     ML Score: {decision.ml_risk_score:.3f}")
        print(f"     Blended Score: {decision.blended_risk_score:.3f} → {decision.blended_classification}")
        print(f"     Risk Zone: {decision.risk_zone.value}")
        print(f"     ML Influenced: {decision.ml_influenced}")
        print(f"     Classification Changed: {decision.classification_changed}")
        print(f"     Explanation: {decision.explanation}")
    
    # Simulate realistic workflow
    print("\n3. Simulating Realistic Decision Workflow...")
    
    for i in range(100):
        agent_id = f"agent_{i % 10}"
        action_id = f"action_{i}"
        
        # Generate features with varying risk levels
        if i < 40:
            risk_level = 'low'
            target = random.uniform(0.1, 0.35)
        elif i < 80:
            risk_level = 'medium'
            target = random.uniform(0.4, 0.6)
        else:
            risk_level = 'high'
            target = random.uniform(0.65, 0.9)
        
        features = simulate_action_features(risk_level)
        rule_score = target
        ml_score = target + random.uniform(-0.15, 0.15)
        ml_score = max(0.0, min(1.0, ml_score))
        
        rule_class = score_to_classification(rule_score)
        
        decision = blended_engine.compute_blended_risk(
            agent_id=agent_id,
            action_id=action_id,
            rule_risk_score=rule_score,
            rule_classification=rule_class,
            ml_risk_score=ml_score,
            ml_confidence=random.uniform(0.7, 0.95),
            features=features
        )
    
    print(f"✓ Processed {blended_engine.metrics.total_decisions} decisions")
    
    # Show metrics
    print("\n4. Blended Enforcement Metrics...")
    metrics = blended_engine.get_metrics_report()
    
    print(f"\n   Zone Distribution:")
    zone_dist = metrics['zone_distribution']
    print(f"   - Clear Allow: {zone_dist['clear_allow']} ({zone_dist['clear_allow']/metrics['total_decisions']*100:.1f}%)")
    print(f"   - Gray Zone: {zone_dist['gray_zone']} ({zone_dist['gray_zone_percentage']:.1f}%)")
    print(f"   - Clear Deny: {zone_dist['clear_deny']} ({zone_dist['clear_deny']/metrics['total_decisions']*100:.1f}%)")
    
    print(f"\n   ML Influence (in Gray Zone):")
    ml_inf = metrics['ml_influence']
    print(f"   - Influenced Decisions: {ml_inf['influenced_count']}")
    print(f"   - Influence Rate: {ml_inf['influence_rate']:.1f}%")
    print(f"   - Classification Changes: {ml_inf['classification_changes']}")
    print(f"   - Change Rate: {ml_inf['change_rate']:.1f}%")
    print(f"   - Escalations (stricter): {ml_inf['escalations']}")
    print(f"   - De-escalations (lenient): {ml_inf['de_escalations']}")
    
    # Gate check
    print("\n5. Promotion Gate Check...")
    gate_metrics = metrics['gate_metrics']
    print(f"\n   Gate Criteria:")
    print(f"   - FP Delta: {gate_metrics['fp_delta']} ({gate_metrics['fp_delta_percentage']:.1f}%)")
    print(f"   - Detection Improvement: +{gate_metrics['detection_improvement']}")
    print(f"   - Passes Gate: {'✓ YES' if gate_metrics['passes_gate'] else '✗ NO'}")
    print(f"   - Reason: {gate_metrics['gate_reason']}")
    
    return blended_engine


def demo_audit_trail():
    """Demonstrate pre/post decision audit trail."""
    print_section("Pre/Post Decision Audit Trail")
    
    print("Creating blended engine with audit logging...\n")
    engine = MLBlendedRiskEngine(
        gray_zone_lower=0.4,
        gray_zone_upper=0.6,
        rule_weight=0.7,
        ml_weight=0.3,
        storage_path="./demo_data/phase6_audit"
    )
    
    # Create decision with change
    print("1. Decision in Gray Zone (ML influences outcome)...")
    
    features = simulate_action_features('medium')
    rule_score = 0.52
    ml_score = 0.68
    rule_class = "warn"
    
    decision = engine.compute_blended_risk(
        agent_id="audit_agent_001",
        action_id="audit_action_001",
        rule_risk_score=rule_score,
        rule_classification=rule_class,
        ml_risk_score=ml_score,
        ml_confidence=0.88,
        features=features
    )
    
    print(f"\n   BEFORE (Rule-based only):")
    print(f"   - Risk Score: {decision.rule_risk_score:.3f}")
    print(f"   - Classification: {decision.rule_classification}")
    
    print(f"\n   ML Input:")
    print(f"   - Risk Score: {decision.ml_risk_score:.3f}")
    print(f"   - Confidence: {decision.ml_confidence:.3f}")
    
    print(f"\n   AFTER (Blended):")
    print(f"   - Risk Score: {decision.blended_risk_score:.3f}")
    print(f"   - Classification: {decision.blended_classification}")
    print(f"   - Formula: {decision.rule_weight}*{decision.rule_risk_score:.3f} + {decision.ml_weight}*{decision.ml_risk_score:.3f} = {decision.blended_risk_score:.3f}")
    
    print(f"\n   Audit Trail:")
    print(f"   - Decision ID: {decision.decision_id}")
    print(f"   - Timestamp: {decision.timestamp}")
    print(f"   - Classification Changed: {decision.classification_changed}")
    print(f"   - Explanation: {decision.explanation}")
    
    # Export for audit
    print("\n2. Exporting Audit Trail...")
    decisions = engine.export_decisions(ml_influenced_only=True, limit=5)
    print(f"✓ Exported {len(decisions)} ML-influenced decisions")
    
    if decisions:
        print(f"\n   Sample Audit Record:")
        audit = decisions[0]
        print(f"   - Agent: {audit['agent_id']}")
        print(f"   - Original: {audit['original_classification']} ({audit['rule_risk_score']:.3f})")
        print(f"   - Final: {audit['final_classification']} ({audit['blended_risk_score']:.3f})")
        print(f"   - Changed: {audit['classification_changed']}")


def demo_gray_zone_analysis():
    """Demonstrate gray zone-specific analysis."""
    print_section("Gray Zone Analysis")
    
    print("Analyzing ML impact in uncertain decisions...\n")
    
    engine = MLBlendedRiskEngine(
        gray_zone_lower=0.4,
        gray_zone_upper=0.6,
        rule_weight=0.7,
        ml_weight=0.3
    )
    
    # Generate gray zone decisions
    print("1. Generating decisions in gray zone (0.4-0.6)...")
    
    for i in range(50):
        # Force into gray zone
        rule_score = random.uniform(0.4, 0.6)
        ml_score = random.uniform(0.3, 0.7)
        
        rule_class = score_to_classification(rule_score)
        features = simulate_action_features('medium')
        
        engine.compute_blended_risk(
            agent_id=f"gray_agent_{i % 5}",
            action_id=f"action_{i}",
            rule_risk_score=rule_score,
            rule_classification=rule_class,
            ml_risk_score=ml_score,
            ml_confidence=random.uniform(0.75, 0.95),
            features=features
        )
    
    # Analyze gray zone
    print(f"✓ Generated {engine.metrics.gray_zone_count} gray zone decisions\n")
    
    gray_zone_decisions = engine.export_decisions(risk_zone=RiskZone.GRAY_ZONE)
    
    print(f"2. Gray Zone Analysis:")
    print(f"   - Total in gray zone: {len(gray_zone_decisions)}")
    
    changed = sum(1 for d in gray_zone_decisions if d['classification_changed'])
    print(f"   - Classifications changed: {changed} ({changed/len(gray_zone_decisions)*100:.1f}%)")
    
    escalated = sum(1 for d in gray_zone_decisions 
                    if d['blended_risk_score'] > d['rule_risk_score'])
    print(f"   - Risk escalated by ML: {escalated}")
    
    de_escalated = sum(1 for d in gray_zone_decisions 
                       if d['blended_risk_score'] < d['rule_risk_score'])
    print(f"   - Risk de-escalated by ML: {de_escalated}")
    
    print(f"\n   This is where ML adds value - helping with uncertain cases!")


if __name__ == "__main__":
    # Run blended risk demo
    engine = demo_blended_risk()
    
    # Run audit trail demo
    demo_audit_trail()
    
    # Run gray zone analysis
    demo_gray_zone_analysis()
    
    print("\n" + "="*60)
    print("Phase 6 Demo Complete!")
    print("="*60)
    print("\nKey Achievements:")
    print("✓ ML blending only in gray zone (uncertain decisions)")
    print("✓ Clear allow/deny zones use rules only")
    print("✓ Complete audit trail of ML influence")
    print("✓ Gate checks ensure controlled rollout")
    print("\nNext Steps:")
    print("- Monitor FP delta in production")
    print("- Validate detection improvements")
    print("- Adjust gray zone boundaries if needed")
    print("- Prepare for Phase 7 (Anomaly & Drift Detection)")
