from setuptools import setup, find_packages

setup(
    name="servo_trader",
    version="0.1.0",
    description="Automated crypto trading bot using regression models and dynamic limit selling.",
    author="Jarred Deluca",
    author_email="jarred.g.deluca@student.uts.edu.au",
    license="MIT",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "requests",
        "scikit-learn",
        "xgboost",
        "catboost",
        "pyyaml",
        "python-binance",
        "alpaca-trade-api"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)