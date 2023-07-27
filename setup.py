import setuptools

version_namespace = {}
with open("crystalpy/version.py") as f:
    exec(f.read(), version_namespace)


setuptools.setup(
    name="crystalpy",
    version=version_namespace["__version__"],
    author="Piotr Jarosik",
    author_email="pjarosik@ippt.pan.pl",
    packages=setuptools.find_packages(exclude=[]),
    classifiers=[
        "Development Status :: 1 - Planning",

        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",

        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.19.3",
        "scipy>=1.3.1"
    ],
    python_requires='>=3.7'
)