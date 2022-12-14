from setuptools import setup, find_packages


EXTRAS_REQUIRE = {
    "dev": [
        "pytest",
        "flake8"
    ],
}
INSTALL_REQUIRES = [
    "ipykernel>=6.14.0",
    "ipython_genutils>=0.2.0",
    "joblib>=1.2",
    "jupyterlab>=3",
    "ipywidgets>=6",
    "matplotlib>=3.6",
    "numpy>=1.21",
    "pandas>=1.2.0; python_version>='3.9'",
    "pandas>=1.0.5; python_version<'3.9'",
    "pyyaml>=6",
    "seaborn>=0.12",
    "scikit-learn>=1.1"
]

setup(
    name="zeuspy",
    python_requires=">=3.8",
    packages=find_packages(),
    package_data={},
    version='0.0.0',
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    zip_safe=False,
    include_package_data=True
)
