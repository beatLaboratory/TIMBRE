# TIMBRE
### ***T***racking ***I***nformative ***M***ultivariate ***B***rain ***R***hythms ***E***fficiently
<img src="Block%20Diagram.svg" width="500" alt="TIMBRE Block Diagram">

TIMBRE is a complex-valued neural network that can recover patterns from multi-channel oscillations carrying information about a variable of interest. It was primarily developed and tested using multi-electrode recordings of local field potentials (LFPs) measured in the hippocampus of rats. 

## Notable Files
1. **TIMBRE.py** - neural network implementation of carrier-free decoding.
2. **toy_data_demo.ipynb** - Jupyter notebook demonstrating TIMBRE on toy data.
3. **LFP_demo.ipynb** - Jupyter notebook demonstrating TIMBRE on real data.
4. **helpers.py** - helper functions for preprocessing data.

Notebooks were tested on Google Colab.

## Recipes for success
Here are some tips that I found helpful when using TIMBRE.
1. **Shrink the data.** Speed up training by decimating data to ~2.5x the sampling rate of your oscillation of interest. For example, the hippocampus data was collected at 1250 Hz, but we were interested in a ~10 Hz signal, so we downsampled the data to 25 Hz. Note that you should apply a low-pass filter before downsampling to avoid aliasing (this is automatically done by python and matlab's `decimate` function).
2. **Use many measurement channels.** For brain data, using as many measurement channels as possible improves accuracy. Note that this is true even if the data appears to be highly correlated across all channels. We use ~250 electrodes and expect further improvements by using more electrodes.
3. **Remove extraneous frequencies.** Brain data contains high power at low frequencies ('1/f noise'). It can help to remove these using a high-pass filter with a cutoff frequency below that of your signal.
4. **Whiten the data.** Use the `whiten` function in `helpers.py` to decorrelate the data across channels.
5. **Balance training data.** Try to evenly represent the different labels within your training data.
6. **Analyze each behavioral condition separately.** Try not to mix data from different behavioral conditions. For example, we apply TIMBRE on periods of running or staying separately, as the oscillations found in these two conditions are very different from one another.
7. **Try different hidden layer sizes.** This is the main hyperparameter. It helps to loop through these and compare performance on held-out test data to chance.
8. **Use coarse-grained labels for training.** If you aren't sure what variables might be represented in your data, identify high-level categories that were measured or manipulated while collecting data. For example, although we collected the rat's position within the maze as a continuous variable, we trained the network to simply predict which of 3 areas the animal was in.
9. **Use fine-grained labels for visualization.** Once you see above-chance performance, visualize the activations of the hidden nodes relative to variables that are hypothesized to be relevant. For example, the hippocampus is known to care about the animal's location, so we visualized hidden node activations as a function of position (even though we trained the network using coarse-grained labels).

## References
Agarwal, B Lustig, S Akera, E Pastalkova, AK Lee, FT Sommer. [News without the buzz: reading out weak theta rhythms in the hippocampus.](https://www.biorxiv.org/content/10.1101/2023.12.22.573160v1) bioRxiv, 2023.12. 22.573160
