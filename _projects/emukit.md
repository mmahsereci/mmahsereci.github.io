---
layout: page
title: EmuKit
description: >
  EmuKit is a highly adaptable Python toolkit for decision-making under uncertainty.
img: assets/img/project_thumbnails/emukit_taxisim.png
importance: 10
category: software
---

EmuKit is a highly adaptable Python toolkit for decision-making under uncertainty. 
It's main packages provide methods for Bayesian optimization, Bayesian quadrature and experimental design.
The core package contains and active learning loop that unifies functionality of those methods.

Emukit's main features is that is it backend agnostic to the modeling backend. Hence, custom models in the Python
ecosystem, can be wrapped into Emukit's interfaces and make use of active data selection. 
This is particularly useful if it is not possible to port the custom model to another library.

Further, Emukit is composed of classes and interfaces that mimic the practical choices of a machine learning practitioner.
Hence, it is possible to design customize methods quite easily, by switching out or adding new components that seamlessly
integrate with the rest of the codebase (plug-and-play).

I am one of the original authors and a co-maintainer of EmuKit.


### Resources

- [GitHub page](https://github.com/EmuKit/emukit) 
- [Webpage](https://emukit.github.io/) 


### Citation

If you use EmuKit in your work, please cite as:

```buildoutcfg
@article{emukit2023,
  author = {A. Paleyes and M. Mahsereci and N.~D. Lawrence},
  title = {Emukit: A Python toolkit for decision making under uncertainty},
  journal = {Proceedings of the 22nd Python in Science Conference},
  pages = {68 - 75},
  year = {2023}
}

@inproceedings{emukit2019,
  author = {Paleyes, A. and Pullin, M. and Mahsereci, M. and Lawrence, N. and Gonzalez, J.},
  title = {Emulation of physical processes with Emukit},
  booktitle = {Second Workshop on Machine Learning and the Physical Sciences, NeurIPS},
  year = {2019},
}
```
