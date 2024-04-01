from distutils.core import setup
from setuptools import find_packages

setup(
    name="keras-explainability",
    version="1.0.0",
    author="Esten Høyland Leonardsen",
    author_email="estenleonardsen@gmail.com",
    packages=find_packages(),
    url="https://github.com/estenhl/keras-explainability",
    install_requires=[
        "pytest",
        "tensorflow"
    ]
)
