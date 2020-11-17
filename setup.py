import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gbm_org_analysis",
    version="0.0.12",
    author="Matt Rybin",
    author_email="mxr2011@miami.edu",
    description="Extracts and analyzes GBM-organoid data exported from Imaris",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rybinmj/gbm_org_analysis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
