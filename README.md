# PyTCI

A toolbox to analyze temporal context invariance of pytorch models.

## Method

The temporal-context invariance paradigm is used to estimate the size of the stimulus window within which stimuli alter the response, and outside of which they have little impact. The method is introduced for biological and artificial systems in the following papers:
1. <a href="https://www.biorxiv.org/content/10.1101/2020.09.30.321687v2">Multiscale integration organizes hierarchical computation in human auditory cortex</a><br/>
2. <a href="https://neurips.cc">Understanding Adaptive, Multiscale Temporal Integration In Deep Speech Recognition Systems</a>

Brief description of the paradigm coming soon... In the meantime, refer to the referenced papers for a full description of the method and its potential applications.

## Usage

To demonstrate the application of the toolbox, two Jupyter notebooks have been provided in the <a href="https://github.com/naplab/PyTCI/tree/main/Examples">`Examples/`</a> directory:

1. <a href="https://github.com/naplab/PyTCI/blob/main/Examples/Example-Toy.ipynb"><strong>Basic</strong></a>: A series of basic examples demonstrating how to use different features of the toolbox on a simple toy model.

2. <a href="https://github.com/naplab/PyTCI/blob/main/Examples/Example-DeepSpeech.ipynb"><strong>Advanced</strong></a>: Applying the toolbox to a complicated PyTorch model and analyzing activations at different layers of the model. This notebook requires the pretrained model and speech audio clips in `Examples/resources.tar` to be extracted and placed in a directory named `Examples/resources`.

The toolbox is intended to be easy-to-use but customizable, so different parts of the method can be swapped out. Refer to the method descriptions for a list of available customizations.

<strong>NOTE</strong>: These notebooks might not render as intended on GitHub. For correct rendering, open them locally in a Jupyter notebook.

## Installation

To install this package through pip, run the following command:

`pip install git+https://github.com/naplab/PyTCI.git`

You can use the same command to update the package.
