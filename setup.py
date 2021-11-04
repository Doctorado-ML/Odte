import setuptools


def readme():
    with open("README.md") as f:
        return f.read()


def get_data(field):
    item = ""
    with open("odte/__init__.py") as f:
        for line in f.readlines():
            if line.startswith(f"__{field}__"):
                delim = '"' if '"' in line else "'"
                item = line.split(delim)[1]
                break
        else:
            raise RuntimeError(f"Unable to find {field} string.")
    return item


setuptools.setup(
    name="Odte",
    version=get_data("version"),
    license=get_data("license"),
    description="Oblique decision tree Ensemble",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/doctorado-ml/odte",
    author=get_data("author"),
    author_email=get_data("author_email"),
    keywords="scikit-learn oblique-classifier oblique-decision-tree decision-\
    tree ensemble svm svc",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    install_requires=["stree"],
    test_suite="odte.tests",
    zip_safe=False,
)
