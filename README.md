# PyTCI

A toolbox that estimates the integration window of a sensory response using the "Temporal Context Invariance" paradigm (TCI).

## The TCI method

Integration windows are defined as the time window within which stimuli alter a sensory response and outside of which stimuli have little effect. Integration windows provide a simple and general way to define the analysis timescale of a response. We estimate integration windows by presenting segments of natural stimuli in two different pseudorandom orders, such that the same segment occurs in two different contexts (is surrounded by different segments). We then estimate the smallest segment duration outside of which stimuli have little effect on the response. The TCI paradigm was initially developed to estimate integration windows for biological neural systems:

1. <a href="https://www.biorxiv.org/content/10.1101/2020.09.30.321687v2">Multiscale integration organizes hierarchical computation in human auditory cortex</a><br/>

The method however can be applied to any sensory response, and we have recently used the method to understand how deep speech recognition systems learn to flexibly integrate across multiple timescales:

2. <a href="https://openreview.net/pdf?id=h4es0CIohF">Understanding Adaptive, Multiscale Temporal Integration In Deep Speech Recognition Systems</a>

This toolbox implements the analyses described in the above NeurIPS paper. We estimate context invariance using the "cross-context correlation" and then estimate the integration window by finding the smallest segment duration needed to achieve a given correlation threshold. Note that this approach is not robust to data noise and thus is not appropriate for biological neural systems (we will be releasing a different toolbox soon that addresses this limitation). 

## Usage

To demonstrate the application of the toolbox, two Jupyter notebooks have been provided in the "Examples" directory:

1. <a href="https://nbviewer.org/github/naplab/PyTCI/blob/main/Examples/Example-Toy.ipynb"><strong>Example-Toy</strong></a>: Shows how to apply the TCI method to a toy model that integrates sound energy within a gamma-distributed window. Covers most of the functionality of the toolbox.

2. <a href="https://nbviewer.org/github/naplab/PyTCI/blob/main/Examples/Example-DeepSpeech.ipynb"><strong>Example-DeepSpeech</strong></a>: Shows how to use the TCI method to estimate integration windows from the DeepSpeech2 model described in the paper, implemented in PyTorch. This notebook requires the pretrained model and speech audio clips in `Examples/resources.tar` to be extracted and placed in a directory named `Examples/resources`. It also has extra dependencies that need to be installed.

<strong>NOTE</strong>: These notebooks might not render as intended on GitHub. For correct rendering, open them locally in a Jupyter notebook or using nbviewer.

## Installation

To install or update this package through pip, run the following command:

`pip install git+https://github.com/naplab/PyTCI.git`
