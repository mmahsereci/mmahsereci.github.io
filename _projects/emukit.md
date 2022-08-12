---
layout: page
title: EmuKit
description: >
  EmuKit is a highly adaptable Python toolkit for decision-making under uncertainty.
img: assets/img/project_thumbnails/emukit_taxisim.png
importance: 10
category: software
---

EmuKit is a highly adaptable Python toolkit for decision-making under uncertainty. Its core components is an 
active learning loop that unifies several active machine learning methods such as experimental design, 
Bayesian optimization and Bayesian quadrature. 
EmuKit's design allows the user to customize the learning algorithm easily, 
by switching out or adding new components (plug-and-play). 
Further, EmuKit provides an interface for the surrogate model, such that custom models can be integrated into
the code quickly. 

I am one of the original authors and a co-maintainer EmuKit.


### Resources

- [GitHub page](https://github.com/EmuKit/emukit) 
- [Webpage](https://emukit.github.io/) 
- [Paper](https://ml4physicalsciences.github.io/2019/files/NeurIPS_ML4PS_2019_113.pdf)


### Citation

If you use EmuKit in your work, please cite as:

```buildoutcfg
@inproceedings{emukit19,
  author = {Paleyes, A. and Pullin, M. and Mahsereci, M. and Lawrence, N. and Gonzalez, J.},
  title = {Emulation of physical processes with Emukit},
  booktitle = {Second Workshop on Machine Learning and the Physical Sciences, NeurIPS},
  year = {2019},
}
```
