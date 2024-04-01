from distutils.core import setup

setup(
    name="keras-explainability",
    version="1.0.0",
    author="Esten Høyland Leonardsen",
    author_email="estenleonardsen@gmail.com",
    packages=["explainability", "explainability.model"],
    url="https://github.com/estenhl/keras-explainability",
    install_requires=[
        "pytest",
        "tensorflow"
    ]
)
