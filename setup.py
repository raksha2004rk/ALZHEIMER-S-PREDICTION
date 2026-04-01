from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    This function reads requirements.txt and returns a list of dependencies
    """

    requirements = []

    try:
        with open(file_path) as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.strip() for req in requirements]

            if HYPEN_E_DOT in requirements:
                requirements.remove(HYPEN_E_DOT)

    except FileNotFoundError:
        print("requirements.txt file not found")
        requirements = []

    return requirements


setup(
    name="ai-health-diagnosis",
    version="0.0.1",
    author="Raksha Kadam",
    author_email="rakshakadam2004@gmail.com",
    description="AI-based Alzheimer’s Disease Detection using MRI and Deep Learning",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.8",
)