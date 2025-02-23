import setuptools
from pathlib import Path

with open("README.md", encoding="utf8") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setuptools.setup(
    name='SOAPy-st',
    version='1.0.0',
    author='Cancer system biology lab',
    author_email='wangheqi2021@sinh.ac.cn',
    description='Spatial Omics Analysis in Python',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/LiHongCSBLab/SOAPy',
    python_requires='>=3.9',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3'],
    keywords=sorted(
        [
            "bio-informatics",
            "spatial omics",
            "spatial transcriptomics",
            "spatial data analysis",
        ]),
    packages=setuptools.find_packages(include=["SOAPy-st", "SOAPy-st.*"]),
    install_requires=requirements
)
