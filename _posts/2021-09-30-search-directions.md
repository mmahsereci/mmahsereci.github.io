---
layout:     distill
title:      "Search Directions & Norms"
author:     mmahsereci
snippet:    ""
date:       2021-09-06
thumbnail-small:  "assets/img/blog/story.jpeg"
category:   techblog
tags:       [Optimization, Machine learning]

authors:
  - name: mmahsereci
    url: "https://github.com/mmahsereci"
    affiliations:
      name: University of T&uuml;bingen
---

Search directions for gradient-based optimization can often be derived by minimizing the loss function subject to a norm. 
The norm gives direct insight into the optimizer's assumption on the local structure of the loss landscape. 
In other words, most optimizers perform gradient descent in a non-Euclidean space. 


# The optimization objective

Most of the content is taken from [Mahsereci 2018]() where the relevant sources are cited. 

Consider an optimization problem of the form

$$ 
w^∗ = \operatorname{arg\,min}_w f ( w ),
$$

where $$f: \mathbb{R}^N\rightarrow \mathbb{R}$$, $$w\mapsto f(w)$$ is the objective function, and $$w^*$$ is the minimizer of $$f$$, 
that is a point $$w^∗ \in\mathbb{R}^N$$ where $$f ( w^∗ )\leq f ( w )$$ for all $$w\in\mathbb{R}^N$$. 
The point $$w^*$$ is also called to *solution* of the optimization problem. 
There might be multiple points fulfilling the requirement, in which case any of them is a solution. 
For our purposes, we will further assume that $$f$$ is bounded from below,
and that $$w^∗$$ is attained in $$\mathbb{R}^N$$. We also assume that $$f$$ is at least
once-differentiable everywhere, such that the multi-output gradient
function $$\nabla f : \mathbb{R}^N\mapsto \mathbb{R}^N$$ , $$w \mapsto \nabla f ( w )$$ exists. 
Therefore $$w^∗$$ equals a
point of vanishing slope, i. e., $$\nabla f ( w^∗ ) = 0$$. 

### Local optimization

If the input-dimension $$N$$ is large, we often simplify the task further, 
and assume that any local minimizer, that is a point $$w^∗$$ that
fulfills $$f ( w^∗ ) \leq f ( w )$$ , for all $$w\in\Omega$$, with $$\Omega\subset\mathbb{R}^N$$ a neighborhood
of $$w^∗$$ , is an acceptable solution as well. The latter statement enables
the use of greedy, gradient-based optimizers that do not explore the
whole domain of $$f ( w )$$, but rather focus on exploiting promising areas
towards a local minimum. This is algorithmically and structurally
more appealing and easier to handle, but comes at the expense of
potentially not finding the best solution possible.


($$L ( w )$$ defined in Eq. 28) is usually solved by greedy, iterative optimization 
routines, all of which will only find local minimizers of $$L$$ (we
will assume that this is satisfactory). There are many good textbooks
for an overview on (non-)convex optimization e. g., [100] [16], some
specifically target neural network optimization [81] [48, § 8].
Most iterative optimizers loop over the following subroutines: i) They
initialize $$w$$ randomly as a current best guess for $$w^∗$$ . ii) They approx-imate 
$$L ( w )$$ around $$w_t$$ , usually with a first- or second-order model.
iii) They define a direction of descent $$p_t\in\mathbb{R}^N$$, also called search direc-
tion, based on this model. iv) They set a scalar step size $$\alpha_t\in\mathbb{R}_+$$ 
(also called learning rate in the neural network community) conditioned on
this direction, and, finally, v) They make a step into the scaled descent
direction:

$$
w_{t + 1} = w_t + \alpha_t p_t .
$$

## Gradient Descent

The arguably most basic algorithm that follows the above equation is 
gradient descent (GD) where the search direction 
$$p_t = −\nabla L ( w_t )$$ locally points
into the direction of the steepest path downhill in Euclidean norm,
meaning that for $$\delta\in\mathbb{R}^N$$ , $$\epsilon > 0$$:

$$
\lim_{\epsilon \rightarrow 0}~
\operatorname{arg\,min}_{\delta:\|\delta\| \leq \epsilon} L ( w_t + \delta ) =
\frac{−\nabla L ( w_t )}{\|\nabla L ( w_t )\|}
: = \frac{p_t}{\|p_t\|} 
$$

which yields the update:

$$
w_{t + 1} = w_t − \alpha_t \nabla L_D ( w_t ) .
$$

The underlying model can be thought of as a first-order Taylor expansion 
$$L ( w ) \approx L ( w_t ) + ( w − w_t )^{\intercal} \nabla L ( w_t )$$ 
around the current
location $$w_t$$ , i. e., the tangent plane to $$L ( w )$$ at $$w_t$$ . This also means that
the local approximation is unbounded below, and does not provide a
meaningful estimate for the step size $$\alpha_t$$. 

Nevertheless,
gd is computationally inexpensive such that many iterations can be
performed; it is quite robust meaning that it is applicable to many
problems as well as numerically stable, and also incredibly easy to
implement (all you need is the gradient). These characteristics make
gd still one of the most widely used iterative optimization routines,
since more than 150 years.

## Newton

What if H is not pos def?