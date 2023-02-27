import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vapor",
    version="1.0.0",
    author="Fereidoon Zangeneh",
    author_email="efreidun@gmail.com",
    description="A Probabilistic Framework for Visual Localization in Ambiguous Scenes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/efreidun/vapor",
    packages=setuptools.find_packages(),
    install_requires=[
        "matplotlib",
        "numpy",
        "open3d",
        "opencv_python",
        "Pillow",
        "PyYAML",
        "scipy",
        "skimage",
        "tikzplotlib",
        "torch",
        "torchvision",
        "tqdm",
    ],
    python_requires=">=3.8",
)
