from setuptools import setup, find_packages

setup(
    name="sqlinjectiondetector",
    version="0.1.0",
    description="A plugin for detecting SQL injection vulnerabilities.",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(include=[
        'cli', 'data', 'audit', 'probes', 'models', 'portal', 'assets', 'deploy',
        'config', 'formal', 'policies', 'training', 'nethical', 'datasets',
        'security', 'dashboards', 'taxonomies', 'governance'
    ]),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
        ]
    },
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
