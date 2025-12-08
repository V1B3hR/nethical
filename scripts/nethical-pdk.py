#!/usr/bin/env python3
"""
Nethical Plugin Development Kit (PDK) CLI

This tool helps developers create, test, and package plugins for Nethical.

Usage:
    nethical-pdk init --name MyDetector --type detector
    nethical-pdk validate ./my-plugin
    nethical-pdk test ./my-plugin
    nethical-pdk package ./my-plugin
    nethical-pdk docs ./my-plugin
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any


class PluginDevelopmentKit:
    """Main PDK class for plugin development operations"""

    PLUGIN_TYPES = ["detector", "policy", "analyzer"]

    DETECTOR_TEMPLATE = '''"""
{name} - Nethical Detector Plugin

{description}

Author: {author}
Version: {version}
"""

from nethical.core.plugin_interface import DetectorPlugin, PluginMetadata
from nethical.detectors.base_detector import SafetyViolation
from typing import List
import logging

logger = logging.getLogger(__name__)


class {class_name}(DetectorPlugin):
    """
    {description}
    """
    
    def __init__(self):
        super().__init__(
            name="{name}",
            version="{version}"
        )
        logger.info(f"Initialized {{self.name}} v{{self.version}}")
    
    async def detect_violations(self, action: Any) -> List[SafetyViolation]:
        """
        Detect violations in the given action.
        
        Args:
            action: The action to analyze
            
        Returns:
            List of detected violations
        """
        violations = []
        
        # TODO: Implement your detection logic here
        # Example:
        # if "sensitive_pattern" in str(action):
        #     violations.append(SafetyViolation(
        #         detector=self.name,
        #         severity="high",
        #         description="Sensitive pattern detected",
        #         category="security",
        #         details={{"pattern": "sensitive_pattern"}}
        #     ))
        
        return violations
    
    @classmethod
    def get_metadata(cls) -> PluginMetadata:
        """
        Return plugin metadata.
        
        Returns:
            PluginMetadata object with plugin information
        """
        return PluginMetadata(
            name="{name}",
            version="{version}",
            description="{description}",
            author="{author}",
            requires_nethical_version=">=0.1.0",
            dependencies=[],
            tags={{"{plugin_type}"}}
        )
    
    async def health_check(self) -> bool:
        """
        Perform health check on the plugin.
        
        Returns:
            True if plugin is healthy, False otherwise
        """
        # TODO: Implement health check logic
        return True
'''

    TEST_TEMPLATE = '''"""
Tests for {name}

Run with: pytest tests/test_{module_name}.py
"""

import pytest
from {module_name} import {class_name}
from nethical.models import AgentAction


@pytest.mark.asyncio
async def test_{snake_name}_initialization():
    """Test plugin initialization"""
    plugin = {class_name}()
    assert plugin.name == "{name}"
    assert plugin.version == "{version}"


@pytest.mark.asyncio
async def test_{snake_name}_detect_violations():
    """Test violation detection"""
    plugin = {class_name}()
    
    # Test with sample action
    action = AgentAction(
        agent_id="test_agent",
        action="test action",
        timestamp="2025-01-01T00:00:00Z"
    )
    
    violations = await plugin.detect_violations(action)
    assert isinstance(violations, list)


@pytest.mark.asyncio
async def test_{snake_name}_metadata():
    """Test plugin metadata"""
    metadata = {class_name}.get_metadata()
    assert metadata.name == "{name}"
    assert metadata.version == "{version}"
    assert metadata.author == "{author}"


@pytest.mark.asyncio
async def test_{snake_name}_health_check():
    """Test plugin health check"""
    plugin = {class_name}()
    healthy = await plugin.health_check()
    assert healthy is True


# Add more tests as needed
'''

    MANIFEST_TEMPLATE = """{{
    "name": "{name}",
    "version": "{version}",
    "description": "{description}",
    "author": "{author}",
    "type": "{plugin_type}",
    "entry_point": "{module_name}.{class_name}",
    "requires_nethical_version": ">=0.1.0",
    "dependencies": [],
    "tags": ["{plugin_type}"],
    "created_at": "{timestamp}",
    "license": "MIT",
    "homepage": "",
    "repository": ""
}}
"""

    README_TEMPLATE = """# {name}

{description}

## Installation

```bash
nethical-pdk install {name}
```

## Usage

```python
from nethical.core import IntegratedGovernance

# Load the plugin
gov = IntegratedGovernance()
gov.load_plugin("{name}")
```

## Configuration

{name} can be configured with the following parameters:

- `parameter1`: Description of parameter 1
- `parameter2`: Description of parameter 2

## Testing

Run tests with:

```bash
pytest tests/
```

## Development

To contribute to this plugin:

1. Clone the repository
2. Install development dependencies: `pip install -e .[dev]`
3. Run tests: `pytest`
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Author

{author}

## Version

{version}
"""

    def __init__(self):
        pass

    def init_plugin(
        self,
        name: str,
        plugin_type: str,
        output_dir: Path,
        description: str = "",
        author: str = "",
    ) -> bool:
        """
        Initialize a new plugin project.

        Args:
            name: Plugin name
            plugin_type: Type of plugin (detector, policy, analyzer)
            output_dir: Output directory for plugin
            description: Plugin description
            author: Plugin author

        Returns:
            True if successful, False otherwise
        """
        if plugin_type not in self.PLUGIN_TYPES:
            print(
                f"Error: Invalid plugin type. Must be one of: {', '.join(self.PLUGIN_TYPES)}"
            )
            return False

        # Create plugin directory structure
        plugin_dir = output_dir / name.lower().replace(" ", "_")

        if plugin_dir.exists():
            print(f"Error: Directory {plugin_dir} already exists")
            return False

        print(f"Creating plugin '{name}' in {plugin_dir}")

        # Create directories
        plugin_dir.mkdir(parents=True, exist_ok=True)
        (plugin_dir / "tests").mkdir(exist_ok=True)
        (plugin_dir / "docs").mkdir(exist_ok=True)

        # Generate file names
        module_name = name.lower().replace(" ", "_").replace("-", "_")
        class_name = "".join(word.capitalize() for word in name.split())
        snake_name = module_name

        # Prepare template variables
        template_vars = {
            "name": name,
            "version": "0.1.0",
            "description": description or f"A custom {plugin_type} plugin for Nethical",
            "author": author or "Plugin Developer",
            "plugin_type": plugin_type,
            "module_name": module_name,
            "class_name": class_name,
            "snake_name": snake_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Create main plugin file
        plugin_file = plugin_dir / f"{module_name}.py"
        with open(plugin_file, "w") as f:
            f.write(self.DETECTOR_TEMPLATE.format(**template_vars))

        print(f"  ✓ Created {plugin_file}")

        # Create test file
        test_file = plugin_dir / "tests" / f"test_{module_name}.py"
        with open(test_file, "w") as f:
            f.write(self.TEST_TEMPLATE.format(**template_vars))

        print(f"  ✓ Created {test_file}")

        # Create manifest
        manifest_file = plugin_dir / "plugin.json"
        with open(manifest_file, "w") as f:
            f.write(self.MANIFEST_TEMPLATE.format(**template_vars))

        print(f"  ✓ Created {manifest_file}")

        # Create README
        readme_file = plugin_dir / "README.md"
        with open(readme_file, "w") as f:
            f.write(self.README_TEMPLATE.format(**template_vars))

        print(f"  ✓ Created {readme_file}")

        # Create __init__.py
        init_file = plugin_dir / "__init__.py"
        with open(init_file, "w") as f:
            f.write(f'"""Nethical {name} Plugin"""\n')
            f.write(f"from .{module_name} import {class_name}\n\n")
            f.write(f'__version__ = "0.1.0"\n')
            f.write(f'__all__ = ["{class_name}"]\n')

        print(f"  ✓ Created {init_file}")

        # Create setup.py
        setup_file = plugin_dir / "setup.py"
        with open(setup_file, "w") as f:
            f.write(
                f"""from setuptools import setup, find_packages

setup(
    name="{name.lower().replace(' ', '-')}",
    version="0.1.0",
    description="{description or f'A custom {plugin_type} plugin for Nethical'}",
    author="{author or 'Plugin Developer'}",
    packages=find_packages(),
    install_requires=[
        "nethical>=0.1.0",
    ],
    extras_require={{
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ]
    }},
    python_requires=">=3.9",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
"""
            )

        print(f"  ✓ Created {setup_file}")

        # Create requirements.txt
        requirements_file = plugin_dir / "requirements.txt"
        with open(requirements_file, "w") as f:
            f.write("nethical>=0.1.0\n")

        print(f"  ✓ Created {requirements_file}")

        # Create .gitignore
        gitignore_file = plugin_dir / ".gitignore"
        with open(gitignore_file, "w") as f:
            f.write("__pycache__/\n*.py[cod]\n*$py.class\n.pytest_cache/\n")
            f.write("dist/\nbuild/\n*.egg-info/\n.venv/\nvenv/\n")

        print(f"  ✓ Created {gitignore_file}")

        print(f"\n✅ Plugin '{name}' created successfully!")
        print(f"\nNext steps:")
        print(f"  1. cd {plugin_dir}")
        print(f"  2. Implement your detection logic in {module_name}.py")
        print(f"  3. Add tests in tests/test_{module_name}.py")
        print(f"  4. Run tests: pytest tests/")
        print(f"  5. Package: nethical-pdk package .")

        return True

    def validate_plugin(self, plugin_path: Path) -> bool:
        """
        Validate a plugin structure and manifest.

        Args:
            plugin_path: Path to plugin directory

        Returns:
            True if valid, False otherwise
        """
        print(f"Validating plugin at {plugin_path}")

        errors = []
        warnings = []

        # Check required files
        required_files = ["plugin.json", "README.md"]
        for filename in required_files:
            file_path = plugin_path / filename
            if not file_path.exists():
                errors.append(f"Missing required file: {filename}")
            else:
                print(f"  ✓ Found {filename}")

        # Validate manifest
        manifest_path = plugin_path / "plugin.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)

                required_fields = [
                    "name",
                    "version",
                    "description",
                    "author",
                    "type",
                    "entry_point",
                ]
                for field in required_fields:
                    if field not in manifest:
                        errors.append(f"Missing required field in plugin.json: {field}")
                    else:
                        print(f"  ✓ Manifest field '{field}': {manifest[field]}")

                # Validate entry point exists
                if "entry_point" in manifest:
                    try:
                        module_name, class_name = manifest["entry_point"].rsplit(".", 1)
                        module_file = plugin_path / f"{module_name}.py"
                        if not module_file.exists():
                            errors.append(
                                f"Entry point module not found: {module_file}"
                            )
                        else:
                            print(f"  ✓ Entry point module exists: {module_file}")
                    except ValueError:
                        errors.append(
                            f"Invalid entry point format: {manifest['entry_point']} (expected 'module.Class')"
                        )

            except json.JSONDecodeError as e:
                errors.append(f"Invalid JSON in plugin.json: {e}")

        # Check for tests
        tests_dir = plugin_path / "tests"
        if not tests_dir.exists():
            warnings.append("No tests directory found")
        else:
            test_files = list(tests_dir.glob("test_*.py"))
            if not test_files:
                warnings.append("No test files found in tests/")
            else:
                print(f"  ✓ Found {len(test_files)} test file(s)")

        # Report results
        if errors:
            print("\n❌ Validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False

        if warnings:
            print("\n⚠️  Warnings:")
            for warning in warnings:
                print(f"  - {warning}")

        print("\n✅ Plugin validation passed!")
        return True

    def test_plugin(self, plugin_path: Path) -> bool:
        """
        Run tests for a plugin.

        Args:
            plugin_path: Path to plugin directory

        Returns:
            True if tests pass, False otherwise
        """
        print(f"Running tests for plugin at {plugin_path}")

        tests_dir = plugin_path / "tests"
        if not tests_dir.exists():
            print("Error: No tests directory found")
            return False

        try:
            result = subprocess.run(
                ["pytest", str(tests_dir), "-v"],
                cwd=plugin_path,
                capture_output=True,
                text=True,
                timeout=60,
            )

            print(result.stdout)
            if result.stderr:
                print(result.stderr)

            if result.returncode == 0:
                print("\n✅ All tests passed!")
                return True
            else:
                print("\n❌ Some tests failed")
                return False

        except subprocess.TimeoutExpired:
            print("Error: Tests timed out")
            return False
        except FileNotFoundError:
            print(
                "Error: pytest not found. Install with: pip install pytest pytest-asyncio"
            )
            return False

    def package_plugin(
        self, plugin_path: Path, output_dir: Optional[Path] = None
    ) -> bool:
        """
        Package a plugin for distribution.

        Args:
            plugin_path: Path to plugin directory
            output_dir: Output directory for package (default: plugin_path/dist)

        Returns:
            True if successful, False otherwise
        """
        print(f"Packaging plugin at {plugin_path}")

        if not self.validate_plugin(plugin_path):
            print("Error: Plugin validation failed. Fix errors before packaging.")
            return False

        output_dir = output_dir or (plugin_path / "dist")
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Build distribution
            result = subprocess.run(
                [sys.executable, "setup.py", "sdist", "bdist_wheel"],
                cwd=plugin_path,
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode == 0:
                print("\n✅ Plugin packaged successfully!")
                print(f"Distribution files in: {plugin_path / 'dist'}")
                return True
            else:
                print("\n❌ Packaging failed:")
                print(result.stderr)
                return False

        except Exception as e:
            print(f"Error: {e}")
            return False

    def generate_docs(self, plugin_path: Path) -> bool:
        """
        Generate documentation for a plugin.

        Args:
            plugin_path: Path to plugin directory

        Returns:
            True if successful, False otherwise
        """
        print(f"Generating documentation for plugin at {plugin_path}")

        manifest_path = plugin_path / "plugin.json"
        if not manifest_path.exists():
            print("Error: plugin.json not found")
            return False

        with open(manifest_path, "r") as f:
            manifest = json.load(f)

        docs_dir = plugin_path / "docs"
        docs_dir.mkdir(exist_ok=True)

        # Generate API documentation
        api_doc = docs_dir / "API.md"
        with open(api_doc, "w") as f:
            f.write(f"# {manifest['name']} API Documentation\n\n")
            f.write(f"Version: {manifest['version']}\n\n")
            f.write(f"## Overview\n\n")
            f.write(f"{manifest['description']}\n\n")
            f.write(f"## Installation\n\n")
            f.write(f"```bash\n")
            f.write(f"pip install {manifest['name'].lower().replace(' ', '-')}\n")
            f.write(f"```\n\n")
            f.write(f"## Usage\n\n")
            f.write(f"```python\n")
            f.write(
                f"from {manifest['entry_point'].rsplit('.', 1)[0]} import {manifest['entry_point'].rsplit('.', 1)[1]}\n\n"
            )
            f.write(f"# Initialize the plugin\n")
            f.write(f"plugin = {manifest['entry_point'].rsplit('.', 1)[1]}()\n\n")
            f.write(f"# Use the plugin\n")
            f.write(f"violations = await plugin.detect_violations(action)\n")
            f.write(f"```\n\n")

        print(f"  ✓ Generated {api_doc}")
        print("\n✅ Documentation generated successfully!")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Nethical Plugin Development Kit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize a new plugin")
    init_parser.add_argument("--name", required=True, help="Plugin name")
    init_parser.add_argument(
        "--type",
        choices=PluginDevelopmentKit.PLUGIN_TYPES,
        default="detector",
        help="Plugin type",
    )
    init_parser.add_argument(
        "--output", type=Path, default=Path("."), help="Output directory"
    )
    init_parser.add_argument("--description", default="", help="Plugin description")
    init_parser.add_argument("--author", default="", help="Plugin author")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate a plugin")
    validate_parser.add_argument("path", type=Path, help="Path to plugin directory")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run plugin tests")
    test_parser.add_argument("path", type=Path, help="Path to plugin directory")

    # Package command
    package_parser = subparsers.add_parser("package", help="Package a plugin")
    package_parser.add_argument("path", type=Path, help="Path to plugin directory")
    package_parser.add_argument("--output", type=Path, help="Output directory")

    # Docs command
    docs_parser = subparsers.add_parser("docs", help="Generate plugin documentation")
    docs_parser.add_argument("path", type=Path, help="Path to plugin directory")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    pdk = PluginDevelopmentKit()

    if args.command == "init":
        success = pdk.init_plugin(
            name=args.name,
            plugin_type=args.type,
            output_dir=args.output,
            description=args.description,
            author=args.author,
        )
        return 0 if success else 1

    elif args.command == "validate":
        success = pdk.validate_plugin(args.path)
        return 0 if success else 1

    elif args.command == "test":
        success = pdk.test_plugin(args.path)
        return 0 if success else 1

    elif args.command == "package":
        success = pdk.package_plugin(args.path, args.output)
        return 0 if success else 1

    elif args.command == "docs":
        success = pdk.generate_docs(args.path)
        return 0 if success else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
