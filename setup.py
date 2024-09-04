from setuptools import setup, find_packages

def readfile(filename):
    with open(filename, 'r+') as f:
        return f.read()

setup(
    name="repo2vec",
    version="0.1.2",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "repo2vec": ["sample-exclude.txt"],
    },
    install_requires=open("requirements.txt").readlines() + ["setuptools"],
    entry_points={
        "console_scripts": [
            "index=repo2vec.index:main",
            "chat=repo2vec.chat:main",
        ],
    },
    author="Julia Turc & Mihail Eric / Storia AI",
    author_email="founders@storia.ai",
    description="A library to index a code repository and chat with it via LLMs.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Storia-AI/repo2vec",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9',
)