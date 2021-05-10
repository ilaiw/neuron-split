# Neuron Splitting with PyTorch
## Introduction
Most of the deep learning neural nets used nowadays are of a fixed size - the number of neurons and layers doesn't change throughout training, only the model weights change.

In this notebook I'd like to explore splitting a single neuron into 2 new ones, at the end of every epoch (the number 2 was arbitrarily selected, as well as splitting at end of epoch). The split is done randomly in such a way that the 2 new neurons are different from one another (essential for the gradient descent), but while keeping the output - or the next layers activations - identical to what they were before the split.

In this way we are expanding the network without changing it in the immediate moment after the split. But of course the hope is that the split will ultimately change the NN's capabilities/potential after additional training.

Below I experiment with a 3 layer dense fully connected NN - the simplest way to try out this idea. The neurons to be split will always be in the hidden layer, since the input and output have fixed sizes defined by the dataset and the number of classes.

[Randomly splitting a neuron with PyTorch notebook](https://nbviewer.jupyter.org/github/ilaiw/neuron-split/blob/main/neuron-split-2.ipynb)

![Neuron split diagram](Neuron-split.jpg)

## Methods
### Experiments
* **No split**: This is the experiment's control. Here the hidden layer size is intialized as the max size which the others will reach. So model starts large as opposed to the rest where starts small and expands with splits.
* **Random split**: A random index in the H dimension is selected to split.
* **Split with max loss check**: Using the validation data, the activations in the hidden layer are iterated over and zeroed one at a time - while checking how thiss affects the overall loss. Basically how is the loss affected if I remove a neuron (similar concept to dropout). So we split the neuron that when removed, validation loss is maximized.
* **Split with min loss check**: Same as above but minimum.
* **Split with loss check probs**: Similar to `Split with max loss check`, but with softmax stochastic selection. Skipping this option for now.
* **Split with max fire rate**: Fire rate is how often an neuron (after ReLU) has a non-zero value. Here we select the neuron with the maximal firing rate for the validation dataset.
* **Split with min fire rate**: Same as previous but minimum.
* **Split with .5 fire rate**: The neuron with a fire rate closest to 0.5 is split.

I ran each experiment 10 times, to see repeatability.

### Number of parameters for each experiment, per epoch
![params-vs-epoch](number-of-params.PNG)

## Results

### Test accuracy - final value
![Test-accuracy-final-value](test-acc-final-val.PNG)

### Test loss - final value
![Test-loss-final-value](test-loss-final-val.PNG)

## Summary
Surpisingly, the split runs were more variable than the fixed size one.
It seems that the neuron splitting works best with half or max firing rate, but other options also have interesting results. The main problem with this type of network is that it introduces more hyperparameters, but it might be useful in certain applications.
The main issue here seems to be when to split and which neuron to split. None of the methods seemed that great, so maybe these decisions need to be made by a separate NN.

Thank you for reading! Feel free to reach out ilaiwai@gmail.com
