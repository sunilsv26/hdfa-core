from setuptools import setup, find_packages

setup(
    name="hdfa-core",
    version="5.0.0",
    author="Sunil Sherikar",
    author_email="sunilsv26@gmail.com",
    description="A non-gradient, cache-native Hyper-Dimensional Fluid Automaton AI core for ultra-low-energy code synthesis.",
    long_description=open("README.md", encoding="utf-8").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(),
    
    # FIXED: Replaced the forbidden license classifier string with the modern SPDX identifier field
    license_expression="Apache-2.0",
    
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "aiohttp>=3.8.0",
        "beautifulsoup4>=4.11.0",
        "streamlit>=1.30.0",
    ],
    entry_points={
        'console_scripts': [
            'hdfa-interactive=hdfa_core.cli:main_entry',
            'hdfa-bridge=hdfa_core.plugin_bridge:main_bridge_entry',
            'hdfa-train=hdfa_core.train_on_repo:main_entry',
        ],
    },
)
