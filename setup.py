from setuptools import setup, find_packages

setup(
    name="hdfa-core",
    version="1.0.0",
    author="Sunil Sherikar",
    author_email="sunilsv26@gmail.com",
    description="A non-gradient, cache-native Hyper-Dimensional Fluid Automaton AI core for ultra-low-energy code synthesis.",
    long_description=open("README.md", encoding="utf-8").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/sunilsv26/hdfa-core", # Update with your active public URL path later
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software Foundation :: Apache License 2.0",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "aiohttp>=3.8.0",
        "beautifulsoup4>=4.11.0",
    ],
)
