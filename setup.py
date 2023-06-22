from setuptools import setup, find_packages

setup(
    name='vocab-coverage',
    version='0.5',
    packages=find_packages(),
    install_requires=[
        'Pillow',
        'protobuf==3.20.0',
        'sentencepiece',
        'tiktoken',
        'torch',
        'transformers',
        'beautifulsoup4',
        'bs4',
        'pandas',
        'requests',
        'scikit-learn',
        'accelerate',
    ],
    description='语言模型中文识字率分析\nA Python package designed to perform coverage analysis on Chinese vocabulary for language models.',
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    author='Tao Wang',
    author_email='twang2218@gmail.com',
    url='https://github.com/twang2218/vocab-coverage',
    license='Apache License 2.0',
    classifiers=[
        'Development Status :: 3 - Alpha',
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    entry_points={
        'console_scripts': [
            'vocab-coverage = vocab_coverage:main',
        ],
    },
)

