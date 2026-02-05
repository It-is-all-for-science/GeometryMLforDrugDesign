from setuptools import setup, find_packages

setup(
    name="geometry-ml-drug-design",
    version="0.1.0",
    description="Geometric Machine Learning for Drug Design Research Project",
    author="DL School Research Team",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torch-geometric>=2.3.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "rdkit>=2022.9.1",
        "gudhi>=3.6.0",
        "pandas>=1.3.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    extras_require={
        "dev": ["pytest>=6.2.0", "pytest-cov>=3.0.0"],
        "notebooks": ["jupyter>=1.0.0", "ipywidgets>=7.6.0"],
        "viz": ["plotly>=5.0.0", "py3Dmol>=2.0.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)