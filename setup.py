from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="eeg-attention-classifier",
    version="1.0.0",
    author="Quantum Brainathon Team",
    author_email="team@quantumbrainathon.com",
    description="EEG-based attention type classification solution for Quantum Brainathon 2025",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quantumbrainathon/eeg-attention-classifier",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "eeg-classifier=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml"],
    },
    extras_require={
        "dev": [
            "pytest>=6.2.5",
            "pytest-cov>=2.12.0",
            "black>=21.5b2",
            "flake8>=3.9.2",
            "isort>=5.9.1",
        ],
    },
)
