from pathlib import Path
from setuptools import setup

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8")

setup(
    name='popcila',
    version='1.0.1',
    author='Youpeng Yang',
    author_email='yypeng1999@gmail.com', 
    url='https://github.com/yyp1999/PopCILA', 
    description='Population-level Complex Phenotypic Intercellular signaling Linkage Analyzer', 
    long_description=README,
    long_description_content_type="text/markdown",
    packages=['PopCILA'],
    package_dir={'PopCILA': 'src'},
    install_requires=[
        'pandas',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'scipy',
        'seaborn',
        'statsmodels',
        'plotly',
        'IPython',
        'scanpy',
        'qnorm',
        'torch',
        'numba',
        'anndata',
        'adjustText',
        'PyComplexHeatmap',
        'libpysal',
        'squidpy',
        'esda',
        'tqdm',
        'networkx',
    ],
    extras_require={
        'chord': ['openchord'],
    },
    classifiers=[
        'Development Status :: 5 - Production/Stable',  
        'Intended Audience :: Science/Research',  
        'License :: OSI Approved :: MIT License',  
        'Programming Language :: Python :: 3', 
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',  
    ],
    python_requires='>=3.8', 
    include_package_data=True,  
    package_data={
        'PopCILA': ['data/*'], 
    },
    zip_safe=False,  
)
