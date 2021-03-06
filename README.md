# parse27k_tools
Tools for the Parse-27k Dataset - evaluation routines and some simple scripts to get started...

Find the dataset here:
[http://www.vision.rwth-aachen.de/parse27k]

## Installation & Requirements
We use Python 2.7 on a 64-bit Linux -- but the tools should run on other platforms as well.

The tools have some dependencies. If you are using Python regularly for scientific computing, 
all of this is likely installed already. Otherwise they can all be easily obtained through `pip`.
These tools depend (directly or indirectly) on:
```
futures
progressbar2
h5py
Pillow
scipy
scikit-image
matplotlib
```
Again, if you have a working Python working environment, most of these will already be installed.


#### Using this in case you have no Python Environment

If you have never used Python before, install Python 2.7 on your system through your favorite installation method (e.g. `apt-get`).
I would then recommend to setup a `virtualenv` with the dependencies. For this, follow these commands:
```
virtualenv new_environment
source new_environment/bin/activate
pip install numpy
pip install scipy
pip install scikit-image
pip install -r requirements.txt
```
*Note* due to some build problems, which I only partially understand, I could not directly install
all requirements with a single `pip` call. It did work, to first install `numpy`,`scipy`, and `scikit-image` separately.
If anybody can point me to a nicer solution, I will be happy to adapt this.

## Usage
You can run the scripts with option `-h` for a basic description of their parameters.
The basic functions are:

* `preprocess_dataset.py`: create crops from the full images and store in `HDF5` file.
* `parse_evaluation.py`: compute performance measures from predictions.
* `visualize_crops.py`: visualize examples from a file generated by `preprocess_dataset.py`.
* `visualize_pid.py`: visualize a specific *pedestrianID* from an `HDF5` file generated by `preprocess_dataset.py`.
The *pedestrianID* is a unique identifier for the examples within the dataset.

## Preprocessing the Dataset
As an example of how to use the dataset, this package includes `preprocess_dataset.py`.
This will read the annotations from the database, crop the examples from the original images, and
store the result in an `HDF5` file.
The resulting file will have two *datasets* (as it is called in `HDF5` language): `crops` and `labels`.
The `labels` is a *NxK* matrix, for *N* examples and *K* attributes.
The meaning is encoded in the `names` attribute.

```
$ export PARSE_PATH=_where_you_saved_the_dataset_
$ ./preprocess_dataset.py --output_dir /tmp/crops --padding_mode zero --padding 32
$ ./visualize_crops.py -r /tmp/crops/train.hdf5
```
## Evaluation Modes
The tool `parse_evaluation.py` allows two modes of evaluation - *retrieval* and *classification*.
See [our paper](http://www.vision.rwth-aachen.de/publications/pdf/parse-acn-iccv2015) for details on the difference.

The tool expects the predictions in an `HDF5`-file, with one *dataset* for each attribute.
You can use the `make_up_predictions.py` script to learn about the expected format.

```
$ ./parse_evaluation.py /tmp/crops/test.hdf5 ./random_predictions.hdf5
```

## Bugs & Problems & Updates
The tools here were taken from a larger set of tools as we prepared the dataset for publication.
In case we notice problems or bugs introduced in this process, we will fix these here.

If you are working with the *Parse-27k* dataset, we encourage you to *follow this repository*.
In case there are any bug-fixes or changes to the evaluation pipeline, GitHub will let you know.

