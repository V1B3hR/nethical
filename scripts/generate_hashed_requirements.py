#!/usr/bin/env python3
"""
Generate hashed requirements files for supply chain security.

This script takes a requirements file and generates a version with SHA256 hashes
for all packages, enabling hash verification during pip install.

Usage:
    python scripts/generate_hashed_requirements.py
    python scripts/generate_hashed_requirements.py --input requirements.txt --output requirements-hashed.txt
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional
import re


class HashedRequirementsGenerator:
    """Generate hashed requirements files with SHA256 verification"""
    
    def __init__(self, input_file: Path, output_file: Optional[Path] = None):
        self.input_file = input_file
        self.output_file = output_file or input_file.parent / f"{input_file.stem}-hashed.txt"
        
    def parse_requirements(self) -> List[Dict[str, str]]:
        """Parse requirements file and extract package information"""
        packages = []
        
        with open(self.input_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                original_line = line
                line = line.strip()
                
                # Preserve comments and empty lines
                if not line or line.startswith('#'):
                    packages.append({
                        'type': 'comment',
                        'content': original_line.rstrip(),
                        'line_num': line_num,
                    })
                    continue
                
                # Parse package==version format
                match = re.match(r'^([a-zA-Z0-9_-]+)==([0-9.]+)', line)
                if match:
                    package_name = match.group(1)
                    version = match.group(2)
                    
                    # Extract inline comment if present
                    comment = ''
                    if '#' in line:
                        parts = line.split('#', 1)
                        comment = '#' + parts[1]
                    
                    packages.append({
                        'type': 'package',
                        'name': package_name,
                        'version': version,
                        'comment': comment,
                        'line_num': line_num,
                    })
                else:
                    # Keep unsupported format as-is
                    packages.append({
                        'type': 'other',
                        'content': original_line.rstrip(),
                        'line_num': line_num,
                    })
        
        return packages
    
    def get_package_hash(self, package_name: str, version: str) -> Optional[str]:
        """Get SHA256 hash for a specific package version"""
        try:
            # Use pip download to get the package and extract hash
            with tempfile.TemporaryDirectory() as tmpdir:
                result = subprocess.run(
                    ['pip', 'download', '--no-deps', '--no-binary', ':all:',
                     f'{package_name}=={version}', '-d', tmpdir],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                
                if result.returncode != 0:
                    # Try with binary if source fails
                    result = subprocess.run(
                        ['pip', 'download', '--no-deps', '--only-binary', ':all:',
                         f'{package_name}=={version}', '-d', tmpdir],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                
                if result.returncode == 0:
                    # Get hash from pip hash command
                    from pathlib import Path
                    downloaded_files = list(Path(tmpdir).glob(f'{package_name}-*'))
                    if downloaded_files:
                        hash_result = subprocess.run(
                            ['pip', 'hash', str(downloaded_files[0])],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )
                        
                        if hash_result.returncode == 0:
                            # Extract SHA256 hash from output
                            for line in hash_result.stdout.split('\n'):
                                if line.startswith(str(downloaded_files[0])):
                                    parts = line.split('--hash=')
                                    if len(parts) > 1:
                                        return parts[1].strip()
        except Exception as e:
            print(f"Warning: Could not get hash for {package_name}=={version}: {e}", file=sys.stderr)
        
        return None
    
    def generate_hashed_requirements(self) -> bool:
        """Generate hashed requirements file"""
        print(f"Generating hashed requirements from {self.input_file}")
        print(f"Output will be written to {self.output_file}")
        
        packages = self.parse_requirements()
        
        with open(self.output_file, 'w') as f:
            # Write header
            f.write("# Hashed requirements file for supply chain security\n")
            f.write("# Generated with SHA256 verification hashes\n")
            f.write("# Install with: pip install --require-hashes -r requirements-hashed.txt\n")
            f.write(f"# Generated from: {self.input_file.name}\n")
            f.write("\n")
            
            for item in packages:
                if item['type'] == 'comment':
                    f.write(item['content'] + '\n')
                elif item['type'] == 'package':
                    package_name = item['name']
                    version = item['version']
                    comment = item['comment']
                    
                    print(f"Processing {package_name}=={version}...", end='', flush=True)
                    
                    # Get hash
                    hash_value = self.get_package_hash(package_name, version)
                    
                    if hash_value:
                        # Write package with hash
                        line = f"{package_name}=={version} \\\n    --hash={hash_value}"
                        if comment:
                            line += f"  {comment}"
                        f.write(line + '\n')
                        print(" ✓")
                    else:
                        # Write package without hash but with warning
                        line = f"{package_name}=={version}  # WARNING: Hash not available"
                        if comment:
                            line += f" {comment}"
                        f.write(line + '\n')
                        print(" ⚠ (hash unavailable)")
                elif item['type'] == 'other':
                    f.write(item['content'] + '\n')
        
        print(f"\nHashed requirements written to {self.output_file}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Generate hashed requirements files for supply chain security'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('requirements.txt'),
        help='Input requirements file (default: requirements.txt)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output hashed requirements file (default: {input}-hashed.txt)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Generate hashed versions of all requirements files'
    )
    
    args = parser.parse_args()
    
    if args.all:
        # Process all requirements files
        files = [
            Path('requirements.txt'),
            Path('requirements-dev.txt'),
        ]
        
        success = True
        for req_file in files:
            if req_file.exists():
                generator = HashedRequirementsGenerator(req_file)
                if not generator.generate_hashed_requirements():
                    success = False
            else:
                print(f"Skipping {req_file} (not found)")
        
        return 0 if success else 1
    else:
        # Process single file
        if not args.input.exists():
            print(f"Error: Input file {args.input} not found", file=sys.stderr)
            return 1
        
        generator = HashedRequirementsGenerator(args.input, args.output)
        return 0 if generator.generate_hashed_requirements() else 1


if __name__ == '__main__':
    sys.exit(main())
