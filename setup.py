import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="molecule-depictor",
    version="1.0.0",
    author="",
    author_email="",
    description="A Python library",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.ibm.com/LUM/molecule-depictor",
    packages=setuptools.find_packages(exclude=["tests.*", "tests"]),
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "License :: Other/Proprietary License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Topic :: Database",
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.7',
    install_requires=[
		"shapely"
    ],
    package_data={"": ["*.json", "*.txt", "*.csv", "*.smi"]},
)