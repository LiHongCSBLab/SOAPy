import setuptools
from pathlib import Path

with open("README.md", encoding="utf8") as readme_file:
    readme = readme_file.read()

setuptools.setup(
    keywords=sorted(
        [
            "bio-informatics",
            "spatial omics",
            "spatial transcriptomics",
            "spatial data analysis",
        ]),
    packages=setuptools.find_packages(include=["SOAPy", "SOAPy.*"]),
)