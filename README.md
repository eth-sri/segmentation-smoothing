Scalable Certified Segmentation via Randomized Smoothing <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>
=============================================================================================================


![img](https://raw.githubusercontent.com/eth-sri/segmentation-smoothing/master/img.png)

We present a new certification method for image and point cloud segmentation based on randomized smoothing. The method leverages a novel scalable algorithm for prediction and certification that correctly accounts for multiple testing, necessary for ensuring statistical guarantees. The key to our approach is reliance on established multiple-testing correction mechanisms as well as the ability to abstain from classifying single pixels or points while still robustly segmenting the overall input. Our experimental evaluation on synthetic data and challenging datasets, such as Pascal Context, Cityscapes, and ShapeNet, shows that our algorithm can achieve, for the first time, competitive accuracy and certification guarantees on real-world segmentation tasks.

For further details, please see [our ICML 2021
paper](https://files.sri.inf.ethz.ch/website/papers/fischer2021segmentation.pdf).


## Setup & Code
To run the code or replicate the experiments, cone the model with submodules and then follow the instructions in [code/README.md](code/README.md):

``` shell
git clone --recurse-submodules https://github.com/eth-sri/segmentation-smoothing.git
cd segmentation-smoothing/code
# follow instructions in code/README.md
```

## Cite

If you use the code in this repository please cite it as:

```
@incollection{fischer2021scalable,
title = {Scalable Certified Segmentation via Randomized Smoothing},
author = { Fischer, Marc and Baader, Maximilian and Vechev, Martin},
booktitle = {International Conference on Machine Learning (ICML)},
year = {2021} }
```

## Contributes
- [Marc Fischer](https://www.sri.inf.ethz.ch/people/marc)
- [Maximilian Baader](https://www.sri.inf.ethz.ch/people/max)
- [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin)




