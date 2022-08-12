---
layout: page
title: ProbNum
description: >
  ProbNum provides numerical solvers for linear systems, intractable integrals and ordinary differential equations in Python.
img: assets/img/project_thumbnails/probnum.png
importance: 10
category: software
---


ProbNum provides numerical solvers for linear systems, intractable integrals and ordinary differential equations in Python.
ProbNum's solvers not only estimate the solution of the numerical problem, but also its uncertainty (numerical error) which 
arises from finite computational resources, discretization and stochastic input. 
The estimated numerical uncertainty can be used in downstream decisions.

Lower level structure of ProbNum includes: A module for random variables and random variable arithmetics;
(memory-)efficient and lazy implementation of linear operators that integrate with random variables;
filtering and smoothing for probabilistic state-space models, mostly variants of Kalman filters.

I am an active contributor and maintainer of ProbNum.

### Resources

- [GitHub page](https://github.com/probabilistic-numerics/probnum) 
- [Webpage](http://pobnum.org) 


### Citation

If you use ProbNum in your work, please cite as:

```buildoutcfg
@misc{probnum21,
    title={ProbNum: Probabilistic Numerics in Python},
    author={J. Wenger and N. Krämer and M. Pf{\"o}rtner and J. Schmidt and N. Bosch and N. Effenberger and J. Zenn and A. Gessner and T. Karvonen and F-X Briol and M. Mahsereci and P. Hennig},
    year = {2021},
    eprint={2112.02100},
    archivePrefix={arXiv},
    primaryClass={cs.MS}
}
```
