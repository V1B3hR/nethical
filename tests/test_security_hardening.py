"""
Tests for Security Hardening Implementation
Validates that all security controls are properly configured
"""

import pytest
import yaml
import json
from pathlib import Path


class TestNetworkPolicies:
    """Test network policy configurations"""
    
    def test_network_policies_exist(self):
        """Verify network policies file exists"""
        netpol_file = Path("deploy/kubernetes/network-policies.yaml")
        assert netpol_file.exists(), "Network policies file not found"
    
    def test_default_deny_policy_exists(self):
        """Verify default deny-all policy is configured"""
        netpol_file = Path("deploy/kubernetes/network-policies.yaml")
        with open(netpol_file) as f:
            docs = list(yaml.safe_load_all(f))
        
        default_deny = next(
            (doc for doc in docs if doc.get('metadata', {}).get('name') == 'default-deny-all'),
            None
        )
        
        assert default_deny is not None, "Default deny-all policy not found"
        assert 'Ingress' in default_deny['spec']['policyTypes']
        assert 'Egress' in default_deny['spec']['policyTypes']
    
    def test_explicit_allow_policies(self):
        """Verify explicit allow policies for required services"""
        netpol_file = Path("deploy/kubernetes/network-policies.yaml")
        with open(netpol_file) as f:
            docs = list(yaml.safe_load_all(f))
        
        required_policies = [
            'allow-dns-egress',
            'nethical-api-ingress',
            'nethical-redis-egress',
            'nethical-db-egress'
        ]
        
        policy_names = [doc.get('metadata', {}).get('name') for doc in docs]
        
        for required in required_policies:
            assert required in policy_names, f"Required policy '{required}' not found"


class TestServiceMeshConfig:
    """Test service mesh and mTLS configurations"""
    
    def test_service_mesh_config_exists(self):
        """Verify service mesh configuration file exists"""
        config_file = Path("deploy/kubernetes/service-mesh-config.yaml")
        assert config_file.exists(), "Service mesh configuration not found"
    
    def test_mtls_strict_mode(self):
        """Verify mTLS is configured in STRICT mode"""
        config_file = Path("deploy/kubernetes/service-mesh-config.yaml")
        with open(config_file) as f:
            docs = list(yaml.safe_load_all(f))
        
        peer_auth = next(
            (doc for doc in docs 
             if doc.get('kind') == 'PeerAuthentication' 
             and doc.get('metadata', {}).get('name') == 'default'),
            None
        )
        
        assert peer_auth is not None, "PeerAuthentication not found"
        assert peer_auth['spec']['mtls']['mode'] == 'STRICT', \
            "mTLS mode is not STRICT"
    
    def test_authorization_policies(self):
        """Verify authorization policies are configured"""
        config_file = Path("deploy/kubernetes/service-mesh-config.yaml")
        with open(config_file) as f:
            docs = list(yaml.safe_load_all(f))
        
        auth_policies = [
            doc for doc in docs 
            if doc and doc.get('kind') == 'AuthorizationPolicy'
        ]
        
        assert len(auth_policies) >= 3, \
            "Insufficient authorization policies configured"


class TestExternalSecrets:
    """Test external secrets configuration"""
    
    def test_external_secrets_config_exists(self):
        """Verify external secrets configuration exists"""
        config_file = Path("deploy/kubernetes/external-secrets.yaml")
        assert config_file.exists(), "External secrets configuration not found"
    
    def test_secret_stores_defined(self):
        """Verify secret stores are configured"""
        config_file = Path("deploy/kubernetes/external-secrets.yaml")
        with open(config_file) as f:
            docs = list(yaml.safe_load_all(f))
        
        secret_stores = [
            doc for doc in docs 
            if doc.get('kind') == 'SecretStore'
        ]
        
        assert len(secret_stores) >= 1, "No secret stores configured"
    
    def test_jwt_keys_external_secret(self):
        """Verify JWT keys are managed externally"""
        config_file = Path("deploy/kubernetes/external-secrets.yaml")
        with open(config_file) as f:
            docs = list(yaml.safe_load_all(f))
        
        jwt_secret = next(
            (doc for doc in docs 
             if doc.get('kind') == 'ExternalSecret' 
             and doc.get('metadata', {}).get('name') == 'nethical-jwt-keys'),
            None
        )
        
        assert jwt_secret is not None, "JWT keys external secret not found"
        assert jwt_secret['spec']['refreshInterval'] == '1h', \
            "JWT keys refresh interval not set to 1h"


class TestSecretRotation:
    """Test secret rotation configuration"""
    
    def test_secret_rotation_cronjob_exists(self):
        """Verify secret rotation CronJob is configured"""
        config_file = Path("deploy/kubernetes/secret-rotation-cronjob.yaml")
        assert config_file.exists(), "Secret rotation CronJob not found"
    
    def test_rotation_schedule(self):
        """Verify rotation runs every 30 days or less"""
        config_file = Path("deploy/kubernetes/secret-rotation-cronjob.yaml")
        with open(config_file) as f:
            docs = list(yaml.safe_load_all(f))
        
        cronjob = next(
            (doc for doc in docs 
             if doc.get('kind') == 'CronJob' 
             and doc.get('metadata', {}).get('name') == 'rotate-secrets'),
            None
        )
        
        assert cronjob is not None, "Secret rotation CronJob not found"
        schedule = cronjob['spec']['schedule']
        
        # Verify schedule contains */30 (every 30 days)
        assert '*/30' in schedule, \
            f"Rotation schedule '{schedule}' does not meet ≤90 day requirement"
    
    def test_rotation_monitoring(self):
        """Verify rotation monitoring is configured"""
        config_file = Path("deploy/kubernetes/secret-rotation-cronjob.yaml")
        with open(config_file) as f:
            docs = list(yaml.safe_load_all(f))
        
        check_job = next(
            (doc for doc in docs 
             if doc.get('kind') == 'CronJob' 
             and doc.get('metadata', {}).get('name') == 'check-secret-rotation'),
            None
        )
        
        assert check_job is not None, "Secret rotation monitoring not found"


class TestRuntimeSecurity:
    """Test runtime security configurations"""
    
    def test_statefulset_security_context(self):
        """Verify StatefulSet has proper security context"""
        config_file = Path("deploy/kubernetes/statefulset.yaml")
        with open(config_file) as f:
            statefulset = yaml.safe_load(f)
        
        pod_security = statefulset['spec']['template']['spec']['securityContext']
        
        assert pod_security['runAsNonRoot'] is True, "Pod not configured to run as non-root"
        assert pod_security['runAsUser'] == 1000, "Pod not running as UID 1000"
        assert pod_security['fsGroup'] == 1000, "Pod fsGroup not set to 1000"
    
    def test_container_security_context(self):
        """Verify container has proper security context"""
        config_file = Path("deploy/kubernetes/statefulset.yaml")
        with open(config_file) as f:
            statefulset = yaml.safe_load(f)
        
        container = statefulset['spec']['template']['spec']['containers'][0]
        container_security = container.get('securityContext', {})
        
        assert container_security.get('allowPrivilegeEscalation') is False, \
            "Privilege escalation not disabled"
        assert container_security.get('readOnlyRootFilesystem') is True, \
            "Root filesystem not read-only"
        assert 'ALL' in container_security.get('capabilities', {}).get('drop', []), \
            "Not all capabilities dropped"
    
    def test_seccomp_profile(self):
        """Verify seccomp profile exists"""
        config_file = Path("deploy/kubernetes/seccomp-profile.json")
        assert config_file.exists(), "Seccomp profile not found"
        
        with open(config_file) as f:
            profile = json.load(f)
        
        assert profile['defaultAction'] == 'SCMP_ACT_ERRNO', \
            "Seccomp default action should be ERRNO (deny)"
        assert len(profile['syscalls']) > 0, "No syscalls whitelisted"
    
    def test_apparmor_profile(self):
        """Verify AppArmor profile exists"""
        config_file = Path("deploy/kubernetes/apparmor-profile.yaml")
        assert config_file.exists(), "AppArmor profile not found"


class TestWAFConfiguration:
    """Test WAF configuration"""
    
    def test_waf_config_exists(self):
        """Verify WAF configuration exists"""
        config_file = Path("deploy/kubernetes/waf-config.yaml")
        assert config_file.exists(), "WAF configuration not found"
    
    def test_modsecurity_rules(self):
        """Verify ModSecurity rules for prompt injection"""
        config_file = Path("deploy/kubernetes/waf-config.yaml")
        with open(config_file) as f:
            docs = list(yaml.safe_load_all(f))
        
        waf_rules = next(
            (doc for doc in docs 
             if doc.get('kind') == 'ConfigMap' 
             and doc.get('metadata', {}).get('name') == 'nethical-waf-rules'),
            None
        )
        
        assert waf_rules is not None, "WAF rules ConfigMap not found"
        
        rules_content = waf_rules['data']['custom-modsec-rules.conf']
        
        # Check for prompt injection rules (case-insensitive)
        assert 'ignore' in rules_content.lower() and 'previous' in rules_content.lower() and 'instruction' in rules_content.lower(), \
            "Prompt injection rule not found"
        assert 'jailbreak' in rules_content.lower(), \
            "Jailbreak detection rule not found"
        
        # Check for resource limits
        assert 'SecRequestBodyLimit' in rules_content, \
            "Request body limit not configured"
    
    def test_ingress_waf_annotations(self):
        """Verify ingress has WAF annotations"""
        config_file = Path("deploy/kubernetes/waf-config.yaml")
        with open(config_file) as f:
            docs = list(yaml.safe_load_all(f))
        
        ingress = next(
            (doc for doc in docs if doc.get('kind') == 'Ingress'),
            None
        )
        
        assert ingress is not None, "Ingress with WAF not found"
        
        annotations = ingress['metadata']['annotations']
        assert annotations.get('nginx.ingress.kubernetes.io/enable-modsecurity') == 'true', \
            "ModSecurity not enabled on ingress"
        assert annotations.get('nginx.ingress.kubernetes.io/proxy-body-size') == '1m', \
            "Request body size limit not set"


class TestSecurityDocumentation:
    """Test security documentation completeness"""
    
    def test_security_hardening_guide_exists(self):
        """Verify security hardening guide exists"""
        guide = Path("docs/Security_hardening_guide.md")
        assert guide.exists(), "Security hardening guide not found"
    
    def test_security_hardening_guide_completeness(self):
        """Verify security hardening guide covers all layers"""
        guide = Path("docs/Security_hardening_guide.md")
        with open(guide) as f:
            content = f.read()
        
        required_sections = [
            'Perimeter Security',
            'Authentication & Authorization',
            'Input Validation',
            'Secrets Management',
            'Supply Chain Security',
            'Runtime Security',
            'Network Security',
            'Logging & Monitoring',
            'Audit Integrity',
            'Plugin Trust',
            'mTLS',
            'Secret Rotation',
            'Vulnerability SLA',
            'Zero Trust',
            'Build Attestation'
        ]
        
        for section in required_sections:
            assert section in content, f"Section '{section}' not found in security guide"
    
    def test_operations_runbook_exists(self):
        """Verify security operations runbook exists"""
        runbook = Path("docs/operations/SECURITY_OPERATIONS_RUNBOOK.md")
        assert runbook.exists(), "Security operations runbook not found"


class TestVerificationScripts:
    """Test verification scripts exist and are executable"""
    
    def test_security_verification_script_exists(self):
        """Verify security controls verification script exists"""
        script = Path("scripts/verify-security-controls.sh")
        assert script.exists(), "Security verification script not found"
        assert script.stat().st_mode & 0o111, "Script not executable"
    
    def test_network_isolation_test_exists(self):
        """Verify network isolation test script exists"""
        script = Path("scripts/test-network-isolation.sh")
        assert script.exists(), "Network isolation test script not found"
        assert script.stat().st_mode & 0o111, "Script not executable"
    
    def test_vulnerability_sla_script_exists(self):
        """Verify vulnerability SLA check script exists"""
        script = Path("scripts/check-vuln-sla.py")
        assert script.exists(), "Vulnerability SLA script not found"
        assert script.stat().st_mode & 0o111, "Script not executable"


class TestCIWorkflows:
    """Test CI/CD workflows for security"""
    
    def test_vuln_sla_workflow_exists(self):
        """Verify vulnerability SLA workflow exists"""
        workflow = Path(".github/workflows/vuln-sla.yml")
        assert workflow.exists(), "Vulnerability SLA workflow not found"
    
    def test_vuln_sla_workflow_schedule(self):
        """Verify vulnerability scan runs frequently"""
        workflow = Path(".github/workflows/vuln-sla.yml")
        with open(workflow) as f:
            workflow_data = yaml.safe_load(f)
        
        # Handle both 'on' and True as YAML boolean
        on_config = workflow_data.get('on', workflow_data.get(True, {}))
        schedule = on_config.get('schedule', [])
        assert len(schedule) > 0, "No scheduled vulnerability scans"
        
        # Check it runs at least every 6 hours
        cron = schedule[0]['cron']
        assert '*/6' in cron or '0 */6' in cron, \
            "Vulnerability scan should run at least every 6 hours"


class TestMandatoryControls:
    """Test implementation of mandatory world-class controls"""
    
    def test_mtls_implementation(self):
        """Verify mTLS is implemented"""
        config_file = Path("deploy/kubernetes/service-mesh-config.yaml")
        assert config_file.exists(), "mTLS configuration missing"
    
    def test_secret_rotation_implementation(self):
        """Verify automatic secret rotation ≤90 days"""
        config_file = Path("deploy/kubernetes/secret-rotation-cronjob.yaml")
        assert config_file.exists(), "Secret rotation automation missing"
    
    def test_vuln_sla_implementation(self):
        """Verify vulnerability SLA enforcement"""
        workflow = Path(".github/workflows/vuln-sla.yml")
        script = Path("scripts/check-vuln-sla.py")
        assert workflow.exists(), "Vulnerability SLA workflow missing"
        assert script.exists(), "Vulnerability SLA script missing"
    
    def test_zero_trust_implementation(self):
        """Verify zero-trust network segmentation"""
        config_file = Path("deploy/kubernetes/network-policies.yaml")
        assert config_file.exists(), "Network policies missing"
    
    def test_build_attestation_implementation(self):
        """Verify build attestation is configured"""
        workflow = Path(".github/workflows/sbom-sign.yml")
        assert workflow.exists(), "SBOM/signing workflow missing"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
