from setuptools import setup, find_packages

setup(
    name="sqlinjectiondetector",
    version="0.1.0",
    description="Detects potential SQL injection patterns in actions",
    author="Nethical Team",
    packages=find_packages(),
    install_requires=[
        "nethical>=0.1.0",
    ],
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
