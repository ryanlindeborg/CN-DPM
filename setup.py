import sys
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
packages = setuptools.find_namespace_packages(include=["cn_dpm*"])
print("PACKAGES FOUND:", packages)
print(sys.version_info)

setuptools.setup(
    name="cn_dpm",
    version="0.0.1",
    author="Ryan Lindeborg",
    author_email="<TODO>",
    description="Import CN-DPM to be used in Sequoia.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="TODO",
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "Method": [
            "cndpm = cn_dpm.cndpm_method:CNDPM",
        ],
    },
    python_requires='>=3.7',
    install_requires=[
        "tensorboardx",
        "simple_parsing>=0.0.15.post1",
    ],
)
