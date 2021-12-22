---
layout: page
title: stochastic optimization
description: >
  High-dimensional stochastic, non-convex optimization such as required to train neural networks.
img: assets/img/project_thumbnails/opt.png
importance: 29
category: research
---

Large datasets yield stochasticity in optimization since only a fraction of the data can be processed at a time.
This yields challenges such as choosing appropriate hyperparameters, a well known one being the learning rate, 
but also finding viable search directions, e.g, by filtering stochastic gradients. 
Additionally, modern optimizers are required to solve talks beyond simply minimizing a given function such as
controlling the generalization gap, or choosing the model itself e.g., via weight pruning. 
It is therefore less clear how the "goodness" of an optimizer can be quantified these days.
In any case, it is still unclear how to best design, retrieve and use the information contained locally at each 
optimization step to simultaneously solve related tasks such as the ones mentioned.

Some aspects of stochastic optimization can be seen as methods of
[probabilistic numerics](https://en.wikipedia.org/wiki/Probabilistic_numerics).

### related open source projects

---

- tbd

### publications

---
- [publication page]({{ site.baseurl }}{% link _pages/publications.md %})
- [PhD thesis](https://publikationen.uni-tuebingen.de/xmlui/handle/10900/84726)