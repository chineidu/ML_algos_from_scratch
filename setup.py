from pathlib import Path

from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt")) as file:
    required_packages = [ln.strip() for ln in file.readlines()]

dev_packages = ["pre-commit==2.19.0"]

style_packages = ["black==22.3.0", "isort==5.10.1", "pylint==2.15.10"]

test_packages = ["pytest==7.1.2", "pytest-cov==2.10.1"]

# Define our package
setup(
    name="ml-from-scratch",
    version=0.1,
    description="Machine Learning Algorithms From Scratch.",
    author="Chinedu Ezeofor",
    author_email="neidu@email.com",
    url="https://github.com/chineidu/ML_algos_from_scratch",
    python_requires=">=3.8",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extras_require={
        "dev": dev_packages + style_packages + test_packages,
        "test": test_packages,
    },
)
