import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="PyTCI",
    version="0.2",
    author="Menoua Keshishian",
    author_email="mk4011@columbia.edu",
    description="Toolbox to analyze temporal context invariance of deep neural networks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/naplab/PyTCI",
    project_urls = {
        "Bug Tracker": "https://github.com/naplab/PyTCI/issues"
    },
    license='MIT',
    packages=['PyTCI'],
    install_requires=[
        'numpy',
        'scipy',
        'torch',
        'torchaudio',
    ],
)
