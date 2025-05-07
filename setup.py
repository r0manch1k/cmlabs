from setuptools import setup, find_packages

setup(
    name="cmlabs",
    version="0.2.0",
    author="Roman Sokolovsky",
    author_email="romansokolv209@gmail.com",
    description="A library for computational mathematics labs",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/r0manch1k/cmlabs",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=["numpy>=1.21.0", "scipy>=1.7.0", "pytest==8.3.5"],
)
