from setuptools import setup

setup(
    name="MxL-GEN",             
    version="0.1.0",
    description="Mechanistic learning for generative engineering package (import as MxL_GEN)",
    author="Elizabeth Hayman",

    packages=[
        "MxL_GEN",
        "MxL_GEN.optimisation",
        "MxL_GEN.surrogate_model",
    ],

    package_dir={
        "MxL_GEN": "src",
        "MxL_GEN.optimisation": "src/optimisation",
        "MxL_GEN.surrogate_model": "src/surrogate_model",
    },
    
    include_package_data=True,
    python_requires='>=3.10.5',
    install_requires=[
        "dask",
        "dask_jobqueue",
        "distributed",
        "dill",
        "deprecated",
        "matplotlib >= 3.9",
        "numpy >= 2",
        "pandas >= 2.2.2",
        "pytest",
        "scikit_learn >= 1.5.2",
        "scipy",
        "seaborn",
        "setuptools",
        "xgboost",
        "tables >= 3.10",
        "paramiko",
        "mkdocs>=1.5.2",
        "mkdocs_material>=9.1.21",
        "mkdocs-drawio-exporter>=0.8.0",
        "mkdocstrings>=0.22.0",
        "mkdocstrings_python>=1.3.0",
        "xgboost",
        "tqdm"
    ],
)
