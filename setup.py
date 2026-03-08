from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip()
        for line in fh
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="smart-manufacturing-ai",
    version="0.1.0",
    author="IEEE IES Industrial AI Lab",
    author_email="industrial-ai-lab@ieee-ies.org",
    description=(
        "AI toolkit for smart manufacturing: defect detection, "
        "robotics anomaly detection, production optimization, and digital twin simulation."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IEEE-IES-Industrial-AI-Lab/Smart-Manufacturing-AI",
    packages=find_packages(exclude=["tests*", "notebooks*", "benchmarks*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "black>=23.0.0", "ruff>=0.1.0"],
    },
    entry_points={
        "console_scripts": [
            "smai-download=datasets.download:main",
        ],
    },
)
