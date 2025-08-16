from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="plant-disease-detection-cnn",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A CNN-based plant disease detection system with 38 disease categories",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/plant-disease-detection-cnn",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "pytest>=6.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
        ],
    },
    keywords="plant disease detection, cnn, deep learning, computer vision, agriculture, tensorflow, keras",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/plant-disease-detection-cnn/issues",
        "Source": "https://github.com/yourusername/plant-disease-detection-cnn",
        "Documentation": "https://github.com/yourusername/plant-disease-detection-cnn#readme",
    },
) 