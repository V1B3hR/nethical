#!/usr/bin/env python3
"""
Nethical CLI - AI Safety Governance System

Main command-line interface for the Nethical AI safety governance system.
Provides top-level automation for scanning, monitoring, and governing AI agents.

Usage:
    python nethical.py --target <agent_id> --scan
    python nethical.py --config config/example_config.yaml
    python nethical.py --target <agent_id> --action "process data" --cohort production
"""

import argparse
import sys
import yaml
import configparser
from pathlib import Path
from typing import Optional, Dict, Any

# Import core Nethical components
try:
    from nethical.core import IntegratedGovernance
    from nethical.core.models import Decision
except ImportError as e:
    print(f"Error: Unable to import Nethical modules: {e}")
    print("Please ensure Nethical is properly installed: pip install -e .")
    sys.exit(1)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration settings
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config if config else {}
    except Exception as e:
        print(f"Error loading YAML config from {config_path}: {e}")
        sys.exit(1)


def load_ini_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from INI file.
    
    Args:
        config_path: Path to INI configuration file
        
    Returns:
        Dictionary containing configuration settings
    """
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        
        # Convert ConfigParser to dict
        result = {}
        for section in config.sections():
            result[section] = dict(config[section])
        return result
    except Exception as e:
        print(f"Error loading INI config from {config_path}: {e}")
        sys.exit(1)


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration file (YAML or INI format).
    
    Args:
        config_path: Path to configuration file (optional)
        
    Returns:
        Dictionary containing configuration settings
    """
    if not config_path:
        return {}
    
    path = Path(config_path)
    if not path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
    
    # Detect file type by extension
    if path.suffix.lower() in ['.yaml', '.yml']:
        return load_yaml_config(config_path)
    elif path.suffix.lower() in ['.ini', '.cfg', '.conf']:
        return load_ini_config(config_path)
    else:
        print(f"Error: Unsupported config file format: {path.suffix}")
        print("Supported formats: .yaml, .yml, .ini, .cfg, .conf")
        sys.exit(1)


def merge_configs(file_config: Dict[str, Any], cli_args: argparse.Namespace) -> Dict[str, Any]:
    """Merge file configuration with CLI arguments.
    
    CLI arguments override file configuration settings.
    
    Args:
        file_config: Configuration loaded from file
        cli_args: Command-line arguments
        
    Returns:
        Merged configuration dictionary
    """
    merged = file_config.copy()
    
    # Override with CLI arguments if provided
    if cli_args.target:
        merged['target'] = cli_args.target
    if cli_args.action:
        merged['action'] = cli_args.action
    if cli_args.cohort:
        merged['cohort'] = cli_args.cohort
    if cli_args.wordlist:
        merged['wordlist'] = cli_args.wordlist
    if cli_args.scan:
        merged['scan'] = True
    if cli_args.storage_dir:
        merged['storage_dir'] = cli_args.storage_dir
        
    return merged


def setup_argparse() -> argparse.ArgumentParser:
    """Set up command-line argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description='Nethical - AI Safety Governance System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan an agent with default settings
  python nethical.py --target agent_123 --scan
  
  # Process a specific action for an agent
  python nethical.py --target agent_456 --action "process user request" --cohort production
  
  # Use configuration file (CLI args override file settings)
  python nethical.py --config config/example_config.yaml
  
  # Use configuration file with CLI override
  python nethical.py --config config/example_config.yaml --target agent_789

For more information, see: https://github.com/V1B3hR/nethical
        """
    )
    
    # Configuration file
    parser.add_argument(
        '--config',
        type=str,
        metavar='FILE',
        help='Path to configuration file (YAML or INI format). CLI arguments override file settings.'
    )
    
    # Core arguments
    parser.add_argument(
        '--target',
        type=str,
        metavar='AGENT_ID',
        help='Target agent ID to scan or monitor'
    )
    
    parser.add_argument(
        '--action',
        type=str,
        metavar='ACTION',
        help='Action description to process through governance'
    )
    
    parser.add_argument(
        '--cohort',
        type=str,
        metavar='COHORT',
        default='default',
        help='Agent cohort for fairness sampling (default: "default")'
    )
    
    parser.add_argument(
        '--scan',
        action='store_true',
        help='Perform a scan of the agent'
    )
    
    parser.add_argument(
        '--wordlist',
        type=str,
        metavar='FILE',
        help='Path to wordlist file for scanning'
    )
    
    # Governance options
    parser.add_argument(
        '--storage-dir',
        type=str,
        metavar='DIR',
        default='./nethical_data',
        help='Storage directory for governance data (default: ./nethical_data)'
    )
    
    parser.add_argument(
        '--enable-all',
        action='store_true',
        help='Enable all governance features'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Nethical 0.1.0'
    )
    
    return parser


def run_scan(governance: IntegratedGovernance, agent_id: str, wordlist: Optional[str] = None, verbose: bool = False):
    """Run a scan on the specified agent.
    
    Args:
        governance: IntegratedGovernance instance
        agent_id: Agent ID to scan
        wordlist: Optional wordlist file path
        verbose: Enable verbose output
    """
    print(f"\n{'='*60}")
    print(f"Scanning Agent: {agent_id}")
    print(f"{'='*60}\n")
    
    # Basic scan - check agent status
    test_actions = [
        "Initialize system",
        "Process user request",
        "Access database",
        "Send email notification"
    ]
    
    if wordlist:
        print(f"Using wordlist: {wordlist}")
        # Load wordlist and use for scanning
        try:
            with open(wordlist, 'r') as f:
                test_actions = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Warning: Could not load wordlist: {e}")
    
    results = []
    for action in test_actions:
        if verbose:
            print(f"Testing action: {action}")
        
        result = governance.process_action(
            agent_id=agent_id,
            action=action,
            cohort='scan'
        )
        
        results.append({
            'action': action,
            'result': result
        })
    
    # Summary
    print(f"\nScan Results for {agent_id}:")
    print(f"{'-'*60}")
    
    for item in results:
        action = item['action']
        result = item['result']
        
        # Extract risk score
        risk_score = result.get('phase3', {}).get('risk_score', 0.0)
        
        print(f"Action: {action[:40]:<40} Risk: {risk_score:.3f}")
    
    print(f"{'-'*60}")
    print(f"Total actions scanned: {len(results)}")
    print()


def run_action(governance: IntegratedGovernance, agent_id: str, action: str, cohort: str, verbose: bool = False):
    """Process a specific action through governance.
    
    Args:
        governance: IntegratedGovernance instance
        agent_id: Agent ID
        action: Action description
        cohort: Agent cohort
        verbose: Enable verbose output
    """
    print(f"\n{'='*60}")
    print(f"Processing Action")
    print(f"{'='*60}")
    print(f"Agent ID: {agent_id}")
    print(f"Cohort: {cohort}")
    print(f"Action: {action}")
    print()
    
    result = governance.process_action(
        agent_id=agent_id,
        action=action,
        cohort=cohort
    )
    
    # Display results
    print("Results:")
    print(f"{'-'*60}")
    
    if 'phase3' in result:
        phase3 = result['phase3']
        print(f"Risk Score: {phase3.get('risk_score', 'N/A')}")
        print(f"Risk Level: {phase3.get('risk_level', 'N/A')}")
    
    if 'phase4' in result and verbose:
        phase4 = result['phase4']
        print(f"Merkle Events: {phase4.get('merkle', {}).get('event_count', 'N/A')}")
    
    if 'phase567' in result:
        phase567 = result['phase567']
        if 'blended' in phase567:
            print(f"Blended Risk: {phase567['blended'].get('blended_risk_score', 'N/A')}")
    
    print(f"{'-'*60}")
    print()


def main():
    """Main CLI entry point."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Load configuration file if provided
    file_config = load_config(args.config)
    
    # Merge configurations (CLI args override file config)
    config = merge_configs(file_config, args)
    
    # Extract governance configuration from file
    gov_config = file_config.get('governance', {}) if file_config else {}
    
    # Build governance initialization parameters
    gov_params = {
        'storage_dir': config.get('storage_dir', './nethical_data'),
        'enable_performance_optimization': gov_config.get('enable_performance_optimization', args.enable_all),
        'enable_merkle_anchoring': gov_config.get('enable_merkle_anchoring', args.enable_all),
        'enable_quarantine': gov_config.get('enable_quarantine', args.enable_all),
        'enable_ethical_taxonomy': gov_config.get('enable_ethical_taxonomy', args.enable_all),
        'enable_sla_monitoring': gov_config.get('enable_sla_monitoring', args.enable_all),
        'enable_shadow_mode': gov_config.get('enable_shadow_mode', args.enable_all),
        'enable_ml_blending': gov_config.get('enable_ml_blending', args.enable_all),
        'enable_anomaly_detection': gov_config.get('enable_anomaly_detection', args.enable_all),
    }
    
    # Add any additional governance config parameters
    for key, value in gov_config.items():
        if key not in gov_params:
            gov_params[key] = value
    
    # Initialize governance system
    try:
        governance = IntegratedGovernance(**gov_params)
        
        if args.verbose:
            print("Governance system initialized successfully.")
        
    except Exception as e:
        print(f"Error initializing governance system: {e}")
        sys.exit(1)
    
    # Determine what to do based on arguments
    target = config.get('target')
    action_text = config.get('action')
    scan = config.get('scan', False)
    
    if not target:
        print("Error: No target agent ID specified. Use --target or specify in config file.")
        parser.print_help()
        sys.exit(1)
    
    # Execute requested operation
    if scan:
        run_scan(governance, target, config.get('wordlist'), args.verbose)
    elif action_text:
        run_action(governance, target, action_text, config.get('cohort', 'default'), args.verbose)
    else:
        print("Error: Must specify either --scan or --action")
        parser.print_help()
        sys.exit(1)
    
    print("Done.")


if __name__ == '__main__':
    main()
