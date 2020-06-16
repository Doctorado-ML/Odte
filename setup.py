import setuptools

__version__ = "0.1.0"
__author__ = "Ricardo Montañana Gómez"


def readme():
    with open("README.md") as f:
        return f.read()


setuptools.setup(
    name="Odte",
    version=__version__,
    license="MIT License",
    description="Oblique decision tree Ensemble",
    long_description=readme(),
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    url="https://github.com/doctorado-ml/stree",
    author=__author__,
    author_email="ricardo.montanana@alu.uclm.es",
    keywords="scikit-learn oblique-classifier oblique-decision-tree decision-\
    tree ensemble svm svc",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    install_requires=["scikit-learn>=0.23.0", "numpy", "ipympl", "stree"],
    test_suite="odte.tests",
    zip_safe=False,
)
