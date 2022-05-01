import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vaincapo",
    version="0.1.0",
    author="Fereidoon Zangeneh",
    author_email="efreidun@gmail.com",
    description="Variational Inference of Camera Pose Posterior Distribution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/efreidun/vaincapo",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy",
        "torch",
    ],
    python_requires=">=3.8",
)
