#setup.py
from setuptools import setup, find_packages

# Read the contents of README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='cryptocurrency-time-series-analysis',
    version='0.1.0',
    author='Harivdan N',
    author_email='harryshastri21@gmail.com',
    description='Advanced Cryptocurrency Time Series Analysis and Prediction',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/harryongit/cryptocurrency-time-series-analysis.git',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.21.0',
        'scikit-learn>=0.24.2',
        'matplotlib>=3.4.2',
        'seaborn>=0.11.1',
        'requests>=2.26.0',
        'tensorflow>=2.6.0',
        'prophet>=1.0.0',
        'python-dotenv>=0.19.0',
        'statsmodels>=0.12.2',
    ],
    extras_require={
        'dev': [
            'pytest>=6.2.4',
            'flake8>=3.9.2',
            'black>=21.6b0',
            'jupyter>=1.0.0',
        ],
        'viz': [
            'plotly>=5.1.0',
            'dash>=2.0.0',
        ]
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords='cryptocurrency time-series analysis machine-learning prediction',
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'crypto-analysis=src.main:main',
        ],
    },
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/cryptocurrency-time-series-analysis/issues',
        'Source': 'https://github.com/yourusername/cryptocurrency-time-series-analysis',
    },
    test_suite='tests',
)
