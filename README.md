# MaatPy

The MaatPy toolbox is written using Python programming language and is deployed as a package. The MaatPy package collects several algorithms that together support an analysis pipeline for imbalanced datasets. It includes novel implementations for:

* SMOTEBoost
* SMOTEBagging
* AdaBoost cost sensitive variants AdaCost, AdaC1, AdaC2 and AdaC3
* Balanced Random Forest Classifier.

as well as modifications for the *imblearn* combination samplers:

* SMOTEENN
* SMOTETomek

## Installation
------------

### Dependencies

scikit-learn requires:
- imbalanced-learn>=0.3.3
- joblib>=0.12.2
- matplotlib>=2.2.2
- numpy>=1.15.0
- pandas>=0.23.4
- scikit-learn>=0.19.2
- scipy>=1.1.0

### User installation

The easiest way to install this package is clone this repository:

```
git clone https://github.com/gkapatai/MaatPy/new/master?readme=1
```

and then run:

```
python setup.py install
```
 This will install the package and all it's dependencies.
 
### Testing

After installation you can launch the test suite by running the following command from within the root directory of this repository:

```
nosetests tests
```
This requires prior installation of the nose python module:
```
pip install nose
```
