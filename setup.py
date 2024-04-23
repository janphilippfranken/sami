from setuptools import find_packages, setup

def get_requirements(path: str):
    return [l.strip() for l in open(path)]

setup(
    name="typo",
    version="0.0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements("requirements.txt"),
)