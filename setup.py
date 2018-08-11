from setuptools import setup, find_packages

setup(
    name='MaatPy',
    version='1.0',
    packages=find_packages(),
    url='https://github.com/gkapatai/MaatPy',
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'imbalanced-learn',
        'matplotlib',
        'pandas',
        'joblib'
    ],
    license='MIT License',
    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Data analysts',
        'Topic :: Software Development :: Classification',

        # Pick your license as you wish (should match "license" above)
        'License ::  MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='classification imbalance',
    author='Georgia Kapatai',
    author_email='gkapatai@gmail.com',
    description='Classification for imbalanced datasets'
)