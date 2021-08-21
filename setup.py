import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="dl-translate",
    version="0.2.4",
    author="Xing Han Lu",
    author_email="github@xinghanlu.com",
    description="A deep learning-based translation library built on Huggingface transformers",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xhlulu/dl-translate",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.6",
    install_requires=[
        "transformers>=4.4.0",
        "torch",
        "sentencepiece",
        "protobuf",
        "tqdm",
    ],
    extras_require={"dev": ["pytest", "black", "jinja2", "mkdocs", "mkdocs-material"]},
)
