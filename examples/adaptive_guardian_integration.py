"""Example integration of Adaptive Guardian with existing nethical modules.

This demonstrates how to integrate the Adaptive Guardian with:
- Detectors
- SafetyJudge
- IntegratedGovernance

The Guardian automatically monitors and adapts security intensity.
"""

import asyncio
import time

from nethical.security.adaptive_guardian import (
    AdaptiveGuardian,
    get_guardian,
    record_metric,
    trigger_lockdown,
    clear_lockdown,
    get_mode,
    get_status,
    monitored,
)
from nethical.security.guardian_modes import GuardianMode


# Example 1: Integrating with a custom detector using the decorator
class MonitoredDetector:
    """Example detector with Guardian monitoring."""
    
    def __init__(self, name: str):
        self.name = name
    
    @monitored("CustomDetector")
    async def detect_violations(self, action):
        """Detect violations with automatic monitoring."""
        # Simulate detection work
        await asyncio.sleep(0.01)
        
        # Your detection logic here
        violations = []
        
        # Return result - Guardian automatically tracks:
        # - Response time
        # - Whether error occurred
        # - Decision made
        return violations


# Example 2: Manual metric recording for fine-grained control
class ManuallyMonitoredJudge:
    """Example judge with manual Guardian integration."""
    
    def __init__(self):
        self.guardian = get_guardian()
    
    async def evaluate_action(self, action):
        """Evaluate action with manual monitoring."""
        start = time.perf_counter()
        error = False
        decision = "ALLOW"
        
        try:
            # Your evaluation logic
            await asyncio.sleep(0.02)
            
            # Make decision
            decision = "ALLOW"  # or "BLOCK", "QUARANTINE", etc.
            
            return {"decision": decision}
        
        except Exception as e:
            error = True
            decision = "ERROR"
            raise
        
        finally:
            # Record metric manually
            response_time_ms = (time.perf_counter() - start) * 1000
            record_metric(
                module="SafetyJudge",
                response_time_ms=response_time_ms,
                decision=decision,
                error=error,
            )


# Example 3: Integration with IntegratedGovernance
class GovernanceWithGuardian:
    """Example governance system with Guardian integration."""
    
    def __init__(self):
        self.guardian = get_guardian()
        self.guardian.start()
    
    @monitored("IntegratedGovernance")
    async def evaluate_with_governance(self, action):
        """Evaluate action through governance with Guardian monitoring."""
        # Check current security mode
        current_mode = get_mode()
        
        # Adjust behavior based on mode
        if current_mode == GuardianMode.SPRINT:
            # Fast path - minimal checks
            return await self._fast_evaluation(action)
        
        elif current_mode == GuardianMode.LOCKDOWN:
            # Full security - all checks
            return await self._full_evaluation(action)
        
        else:
            # Normal evaluation
            return await self._normal_evaluation(action)
    
    async def _fast_evaluation(self, action):
        """Fast evaluation with minimal overhead."""
        await asyncio.sleep(0.005)
        return {"decision": "ALLOW", "mode": "fast"}
    
    async def _normal_evaluation(self, action):
        """Normal evaluation."""
        await asyncio.sleep(0.02)
        return {"decision": "ALLOW", "mode": "normal"}
    
    async def _full_evaluation(self, action):
        """Full evaluation with all security checks."""
        await asyncio.sleep(0.1)
        return {"decision": "BLOCK", "mode": "full"}


# Example 4: Manual lockdown trigger based on external intelligence
class ThreatIntelligenceIntegration:
    """Example integration with external threat intelligence."""
    
    def __init__(self):
        self.guardian = get_guardian()
    
    async def process_threat_intel(self, intel_data):
        """Process threat intelligence and trigger lockdown if needed."""
        threat_level = intel_data.get("threat_level", 0)
        
        if threat_level >= 0.9:
            # High threat detected - trigger manual lockdown
            trigger_lockdown(reason=f"external_threat_intel_level_{threat_level}")
            print(f"🔒 LOCKDOWN triggered: threat level {threat_level}")
        
        elif threat_level < 0.3 and get_mode() == GuardianMode.LOCKDOWN:
            # Threat subsided - clear lockdown
            clear_lockdown()
            print(f"✅ Lockdown cleared: threat level {threat_level}")


# Example 5: Monitoring dashboard / status reporting
class GuardianDashboard:
    """Example dashboard for Guardian status."""
    
    def __init__(self):
        self.guardian = get_guardian()
    
    def print_status(self):
        """Print current Guardian status."""
        status = get_status()
        
        print("\n" + "="*60)
        print("🛡️  ADAPTIVE GUARDIAN STATUS")
        print("="*60)
        
        # Current mode
        mode_info = status
        print(f"\n📊 Current Mode: {mode_info['current_mode']} {mode_info['mode_emoji']}")
        print(f"   Description: {mode_info['mode_description']}")
        print(f"   Duration: {mode_info['mode_duration_s']:.1f}s")
        
        # Manual lockdown
        if mode_info['manual_lockdown']:
            print(f"\n🔒 Manual Lockdown Active: {mode_info['lockdown_reason']}")
        
        # Threat analysis
        threat = mode_info['threat_analysis']
        print(f"\n⚠️  Threat Analysis:")
        print(f"   Overall Score: {threat['overall_score']:.3f}")
        print(f"   Recommended Mode: {threat['recommended_mode']}")
        print(f"   Alert Count: {threat['alert_count']}")
        print(f"   Error Rate: {threat['error_rate']:.1%}")
        print(f"   Response Time Trend: {threat['response_time_trend']}")
        
        if threat['anomaly_modules']:
            print(f"   Anomaly Modules: {', '.join(threat['anomaly_modules'])}")
        
        # Performance
        perf = mode_info['performance']
        print(f"\n⚡ Performance:")
        print(f"   Avg Overhead: {perf['avg_overhead_ms']:.3f}ms (target: {perf['target_overhead_ms']:.3f}ms)")
        print(f"   Max Overhead: {perf['max_overhead_ms']:.3f}ms")
        print(f"   Pulse Interval: {perf['pulse_interval_s']}s")
        
        # Statistics
        stats = mode_info['statistics']
        print(f"\n📈 Statistics:")
        print(f"   Total Metrics: {stats['total_metrics_recorded']}")
        print(f"   Total Alerts: {stats['total_alerts_triggered']}")
        print(f"   Manual Lockdowns: {stats['manual_lockdowns']}")
        print(f"   Automatic Lockdowns: {stats['automatic_lockdowns']}")
        
        # Mode durations
        if stats['mode_durations']:
            print(f"\n⏱️  Time in Each Mode:")
            for mode, duration in stats['mode_durations'].items():
                print(f"   {mode}: {duration:.1f}s")
        
        # Watchdog
        watchdog = mode_info['watchdog']
        print(f"\n👁️  Watchdog:")
        print(f"   Running: {watchdog['running']}")
        print(f"   Guardian Responsive: {watchdog['is_guardian_responsive']}")
        print(f"   Time Since Heartbeat: {watchdog['time_since_heartbeat_s']:.1f}s")
        
        print("\n" + "="*60 + "\n")


# Example 6: Complete integration example
async def complete_integration_example():
    """Complete example showing all integration patterns."""
    
    print("🚀 Starting Adaptive Guardian Integration Example\n")
    
    # Initialize components
    detector = MonitoredDetector("ExampleDetector")
    judge = ManuallyMonitoredJudge()
    governance = GovernanceWithGuardian()
    threat_intel = ThreatIntelligenceIntegration()
    dashboard = GuardianDashboard()
    
    # Simulate normal operation
    print("📝 Simulating normal operation...")
    for i in range(10):
        await detector.detect_violations({"action": f"test_{i}"})
        await judge.evaluate_action({"action": f"test_{i}"})
        await governance.evaluate_with_governance({"action": f"test_{i}"})
        await asyncio.sleep(0.05)
    
    # Show status
    dashboard.print_status()
    
    # Simulate some errors
    print("⚠️  Simulating error conditions...")
    for i in range(5):
        try:
            # Simulate slow response
            record_metric("TestModule", 6000.0, "ALLOW", False)
            await asyncio.sleep(0.1)
        except Exception:
            pass
    
    # Wait for mode adaptation
    await asyncio.sleep(0.5)
    dashboard.print_status()
    
    # Simulate external threat
    print("🚨 Simulating external threat intelligence...")
    await threat_intel.process_threat_intel({"threat_level": 0.95})
    
    await asyncio.sleep(0.2)
    dashboard.print_status()
    
    # Clear threat
    print("✅ Clearing threat...")
    await threat_intel.process_threat_intel({"threat_level": 0.2})
    
    await asyncio.sleep(0.2)
    dashboard.print_status()
    
    print("✨ Integration example complete!\n")


# Example 7: Best practices for integration
class BestPracticesExample:
    """Best practices for integrating Adaptive Guardian."""
    
    @staticmethod
    def integration_checklist():
        """Print integration checklist."""
        print("\n📋 INTEGRATION BEST PRACTICES\n")
        
        practices = [
            "1. Use @monitored decorator for simple monitoring",
            "2. Use record_metric() for fine-grained control",
            "3. Start Guardian early in application lifecycle",
            "4. Monitor Guardian status in dashboards",
            "5. Set up alerts for Guardian watchdog failures",
            "6. Integrate with external threat intelligence",
            "7. Test manual lockdown procedures",
            "8. Monitor performance overhead in production",
            "9. Track mode transitions for capacity planning",
            "10. Use Guardian statistics for security audits",
        ]
        
        for practice in practices:
            print(f"   ✓ {practice}")
        
        print("\n💡 TIPS:\n")
        tips = [
            "- Guardian runs in background thread - no await needed",
            "- Singleton pattern ensures one Guardian per process",
            "- Watchdog runs independently - monitors Guardian itself",
            "- Mode transitions are automatic based on threat score",
            "- Manual lockdown overrides automatic mode selection",
            "- Record metrics for ALL critical operations",
            "- Use correlation tracking for coordinated attacks",
        ]
        
        for tip in tips:
            print(f"   {tip}")
        
        print()


# Main execution
if __name__ == "__main__":
    # Show best practices
    BestPracticesExample.integration_checklist()
    
    # Run complete example
    asyncio.run(complete_integration_example())
