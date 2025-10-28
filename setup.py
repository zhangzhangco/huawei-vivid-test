"""
HDR色调映射专利可视化工具安装脚本
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="hdr-tone-mapping-gradio",
    version="1.0.0",
    author="HDR Research Team",
    author_email="research@hdr-team.com",
    description="基于Gradio框架的HDR色调映射专利可视化工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hdr-team/hdr-tone-mapping-gradio",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "gpu": ["numba>=0.56.0", "cupy>=10.0.0"],
        "exr": ["OpenEXR>=1.3.0"],
        "dev": ["pytest>=6.0.0", "pytest-cov>=3.0.0", "black>=22.0.0", "flake8>=4.0.0"],
    },
    entry_points={
        "console_scripts": [
            "hdr-tone-mapping=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.json", "*.md"],
    },
)