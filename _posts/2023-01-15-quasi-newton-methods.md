---
layout:     post
title:      "Quasi-Newton Methods"
author:     mmahsereci
description:    ""
date:       2023-01-15
category:   techblog
tags:       [optimization, machinelearning]
description: >
    Limited memory BFGS (L-BFGS) is one of the most successful gradient-based optimizers and
    arguably the gold-standard in deterministic, non-convex optimization. 
    It is a member of the Dennis family of quasi-Newton methods that use low-rank approximations of the inverse Hessian to project the gradient. 
    The resulting search direction can thus be thought of as an approximation to the Newton direction, 
    with the important difference that, even for non-convex objective functions, it is always a descent direction. 
    There is a multitude of symmetric and non-symmetric quasi-Newton updates, and here we'll discuss the most relevant ones. 

authors:
  - name: mmahsereci
    url: "https://github.com/mmahsereci"
    affiliations:
      name: University of T&uuml;bingen
---

Limited memory BFGS ([L-BFGS](https://en.wikipedia.org/wiki/Limited-memory_BFGS)) is one of the most successful gradient-based optimizers and
arguably the gold-standard in deterministic, non-convex optimization. 
It is a member of the Dennis family of quasi-Newton methods that use low-rank approximations of the inverse Hessian to project the gradient. 
The resulting search direction can thus be thought of as an approximation to the Newton direction, 
with the important difference that, even for non-convex objective functions, it is always a descent direction. 
There is a multitude of symmetric and non-symmetric quasi-Newton updates, and here we'll discuss the most relevant ones. 

## Optimization objective and notation

Let $$f: \mathbb{R}^d\rightarrow \mathbb{R}$$, $$w\mapsto f(w)$$ be a function that is at least twice
differentiable. 
We aim to solve the optimization problem

$$ 
w^∗ = \operatorname*{arg\,min}_w f (w)
$$

where we are interested in finding the input $$w^*$$ that minimizes $$f$$. 

The gradient function w.r.t $$w$$ is denoted as $$\nabla f: \mathbb{R}^d\rightarrow \mathbb{R}^d$$, $$w\mapsto \nabla f(w)$$
and the Hessian function as  $$\nabla^2 f: \mathbb{R}^d\rightarrow \mathbb{R}^{d\times d}$$, $$w\mapsto \nabla^2 f(w)$$.
Greedy, gradient-based optimizers such as Newton's method generally find local minimizers of $$f$$ which is 
often deemed sufficient.

## Newton's method

We will briefly introduce Newton's method here as quasi-Newton methods aim to approximate the Newton step. 
Newton's method starts with an initial guess $$w_0$$ for $$w^*$$ and then updates the guess by iterating over the following line:

$$
w_{t+1} = w_t - \nabla^2 f(w_t)^{-1} \nabla f(w_t).
$$

Hence, the new guess $$w_{t+1}$$ is found by updating the old guess $$w_t$$ with the vector $$p_t^{\mathrm{newton}}:=-\nabla^2 f(w_t)^{-1} \nabla f(w_t)$$.
We will call $$p_t^{\mathrm{newton}}$$ the *Newton direction* or *Newton step*. 

## Quasi-Newton methods

As the name suggests, quasi-Newton methods aim to approximate the Newton step. 
There is a zoo of quasi-Newton methods out there
which in some sense also reflects the history of discovery of these powerful methods. 
Two famous families are the *Broyden* and the *Dennis* family of quasi-Newton methods the latter of which contains
arguably the most successful quasi-Newton method: [BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm). 
We will have a look at the Dennis family later on.

#### The quasi-Newton step

In essence, quasi-Newton methods replace the Hessian matrix $$\nabla^2 f(w_t)$$ at every step $$t$$
with an estimator $$B_t\in\mathbb{R}^{d\times d}$$ thereof.
The *quasi-Newton step* is then defined as

$$
p_t^{\mathrm{quasi}} = - B_t^{-1}\nabla f(w_t),
$$

where the exact estimator $$B_t = \nabla^2 f(w_t)$$ recovers the Newton step.

In contrast to Newton's method, quasi-Newton methods never evaluate the Hessian function itself but build the
estimate $$B_t$$ from past gradient evaluations $$\nabla f(w_t), \nabla f(w_{t-1}),\dots$$.

How this is done, we will see next.

#### The secant equation

First, we observe that the gradient $$\nabla f(w)$$ and the Hessian $$\nabla^2 f(w)$$ are
related in the sense that the latter is the Jacobien of the former. 
Hence, it is possible to learn about one by evaluating the other.
In particular, quasi-Newton methods estimate the Hessian from past gradient evaluations.
This idea is encoded in the *secant equation*.

Let $$s_t := w_{t+1} - w_t$$ be the path segment in parameter space and 
$$\Delta y_t:= \nabla f(w_{t+1}) - \nabla f(w_t)$$ be the corresponding gradient difference. 

The basic assumption for all quasi-Newton methods is that the estimator $$B_t$$ must fulfill the secant equation

$$
\begin{equation*}
    B_{t} s_{t-1} = \Delta y_{t-1},\quad\text{(secant equation)}
\end{equation*}
$$

which means that $$B_t$$ is equal to the *average* Hessian along the sub-space $$s_{t-1}$$.

This can be seen from the [fundamental theorem of calculus for line integrals](https://en.wikipedia.org/wiki/Gradient_theorem).
In other words, parametrize the path $$\bar{s}(\tau) = w_{t} + \tau s_{t-1}$$ with 
$$\tau\in[0, 1]$$, fulfilling $$\bar{s}(0) = w_{t-1}$$ and $$\bar{s}(1) = w_t$$. 
Then, the average Hessian along the path $$\bar{s}$$ ist

$$
\begin{equation*}
\begin{split}
\int_{\bar{s}} \nabla^2 f(\bar{s})\frac{\mathrm{d}\bar{s}}{\|\bar{s}\|}
&= \int_{0}^{1}\nabla \left[\nabla f(\bar{s}(\tau))\bar{s}'(\tau)\right]\frac{\mathrm{d}\tau}{\|\bar{s}\|} \\
&= \int_{0}^{1}\nabla \left[\frac{\mathrm{d}f(\bar{s}(\tau))}{\mathrm{d}\tau}\right]\frac{\mathrm{d}\tau}{\|\bar{s}\|} \\
&=\nabla \left[f(\bar{s}(1)) - f(\bar{s}(0))\right] \frac{1}{\|\bar{s}\|} \\
& =\left[  \nabla f(w_{t}) - \nabla f(w_{t-1})\right]\frac{1}{\|\bar{s}\|} \\
& =\frac{\Delta y_{t-1}}{\|\bar{s}\|}\\
& =B_t \frac{s_{t-1}}{\|\bar{s}\|},
\end{split}
\end{equation*}
$$

where the last line uses the definition of the secant equation.

We found a way to identify the Hessian estimator $$B_t$$ in the subspace spanned by the path segment 
$$s_{t-1}$$. However, this does not identify $$B_t$$ fully.
Hence, we need further assumptions. In the next section we will explore how the Dennis family
of quasi-Newton methods resolves this issue (Other families take a similar approach, so we will skip them here).

### The Dennis family of quasi-Newton methods

We will now see how the Dennis family of quasi-Newton methods uses the secant equation and 
further assumptions to identify $$B_t$$.

Quasi-Newton methods take an iterative approach to estimating $$B_t$$. This means that we start with an 
initial guess $$B_0$$ for the Hessian at $$w_0$$ (this can for example be a scaled identity matrix $$\sigma_0^2I$$)
and update this guess at every iteration. 

Consider a guess $$B_t$$. 
In the Dennis family of quasi-Newton methods
the subsequent guess $$B_{t+1}$$ for the Hessian matrix, is the solution to the following constrained optimization problem:

$$
\begin{alignat}{2}
\label{eq:qn}
    &&B_{t+1} &= \operatorname{arg\,min}_{B}\|B - B_t\|_{W, F}\\
    &&&\notag \\
    &\text{s.t.}\qquad& Bs_t &= \Delta y_t,   \qquad (\text{secant equation})\notag\\
    &\text{and}\qquad&  B    &= B^{\intercal}.\qquad (B\text{ symmetric})\notag
\end{alignat}
$$

Again, we observe that the first constraint (the secant equation) partially identifies $$B_{t+1}$$ in the sub-space spanned by $$s_t$$.
Further, the second constraint (the symmetry assumption, known to be true for Hessians) restricts the solution to the space of
symmetric matrices. Lastly, to fully identify $$B_{t+1}$$, the minimization problem of Eq. \eqref{eq:qn}
is solved, which, under the mentioned constraints, finds a matrix that is closest to the previous guess $$B_t$$ w.r.t.
the weighted Frobenius norm with weight matrix $$W$$. 
The norm-minimization aspect can be thought of as a regularizer on $$B$$, but most importantly it
identifies $$B_{t+1}$$ in the remaining space that is not covered by the constraints.
All previous path segments $$s_{t-1}$$, $$s_{t-2}$$, $$\dots$$ and corresponding gradient differences are implicitly represented in the 
preceding guess $$B_t$$.

I will not explain the Frobenius norm here (I have another post 
[here]({% link _posts/2022-08-05-kronecker.md %})
where it is mentioned in the context
of Kronecker products). But essentially, given a symmetric positive definite weight matrix $$W\in\mathbb{R}^{d\times d}$$, 
the weighted Frobenius norm
is a matrix norm defined as $$\|X\|_{W, F} = \|W^{\frac{1}{2}}XW^{\frac{1}{2}}\|_F$$ with
$$\|X\|_F^2 = \sum_{i=1}^{d} \sum_{j=1}^{d}X_{ij}^2$$ the (square of the) standard Frobenius norm.
Due to the fact, that $$B_{t+1}$$ is the solution of a norm-minimization problem parametrized by some weight matrix
$$W$$, quasi-Newton methods are also denoted as *variable metric* methods in the literature.

Retrieving $$B_{t+1}$$ from Eq. \eqref{eq:qn} is a bit tedious (proof can be found in the [reference](#references)). 
The important point is that the solution $$B_{t+1}$$ is *analytic*, and, for some vector $$c_t := Ws_t\in\mathbb{R}^d$$ that is 
identified by $$W$$ and $$s_t$$, can be written as

$$
\begin{equation}
\label{eq:dennis}
    B_{t+1} = B_t + \frac{(\Delta y_t - B_ts_t)c_t^{\intercal} + c_t(\Delta y_t - B_ts_t)^{\intercal}}{c_t^{\intercal}s_t} - \frac{c_ts_t^{\intercal}(\Delta y_t - B_ts_t)c_t^{\intercal}}{(c_t^{\intercal}s_t)^2}.
\end{equation}
$$

Eq. \eqref{eq:dennis} describes the Hessian estimates of the *Dennis family* of quasi-Newton methods [[2]](#references) which is 
parameterized by the weight matrix $$W$$ of the metric used and in consequence simply by the vector $$c_t\in\mathbb{R}^d$$. 
All formulas only ever require choosing $$c_t$$ and never $$W$$ explicitly.
For every different $$c_t$$, we obtain another member of the Dennis family.

The most relevant members of the Dennis family are (see [references](#references))

$$
\begin{alignat*}{3}
    &\text{SR1} & c_t & = \Delta y_t - B_ts_t & W_t &= \nabla^2 f(w_t) - B_t\\
    &\text{PSB} & c_t & = s_t & W_t &= I\\
    &\text{Greenstadt}\qquad & c_t & = B_ts_t & W_t &= B_t\\
    &\text{DFP} & c_t & = \Delta y_t & W_t &= \nabla^2 f(w_t)\\
    &\text{BFGS} & c_t & = \Delta y_t + \sqrt{\frac{s_t^{\intercal} \Delta y_t}{s_t^{\intercal} B_t s_t}}B_ts_t\quad  & W_t &= \nabla^2 f(w_t) + \sqrt{\frac{s_t\Delta y_t}{s_t^{\intercal} B_t s_t}}B_t.
\end{alignat*}
$$

In particular, the highly performant BFGS estimator is

$$
\begin{equation*}
    \begin{split}
        B_{t+1} & = B_t + \frac{\Delta y_t \Delta y_t^{\intercal}}{\Delta y_t^{\intercal} s_t} - \frac{B_ts_ts_t^{\intercal} B_t}{s_t^{\intercal} B_ts_t}.
    \end{split}
\end{equation*}
$$

#### Computing the quasi-Newton step
In order to obtain the quasi-Newton update $$p_t^{\mathrm{quasi}}$$, the inverse $$B_t^{-1}$$ is required.
From Eq. \eqref{eq:dennis}, $$B_t^{-1}$$ is easy to obtain via the 
[matrix inversion lemma](https://en.wikipedia.org/wiki/Woodbury_matrix_identity). 

The BFGS estimator for the inverse Hessian $$B_{t+1}^{-1}$$ for example is

$$
\begin{equation}
\label{eq:bfgs}
    \begin{split}
        B_{t+1}^{-1} & = \left(I -\frac{s_t\Delta y_t^{\intercal}}{\Delta y_t^{\intercal} s_t} \right)B_{t}^{-1}\left(I -\frac{\Delta y_ts_t^{\intercal}}{\Delta y_t^{\intercal} s_t} \right) - \frac{s_ts_t^{\intercal}}{\Delta y_t^{\intercal} s_t}\\
        & = B_t^{-1} + \frac{(\Delta y_t^{\intercal} s_t + \Delta y_t^{\intercal} B_t^{-1}\Delta y_t)(s_ts_t^{\intercal})}{(\Delta y_t^{\intercal} s_t)^2} 
        - \frac{B_t^{-1}\Delta y_ts_t^{\intercal} + s_t\Delta y_t^{\intercal} B_t^{-1}}{\Delta y_t^{\intercal}s_t}.
    \end{split}
\end{equation}
$$

Hence, in practice, only ever $$B_t^{-1}$$ needs to be stored which is then updated with a rank-2 term at every iteration.
This removes the need to i) construct $$B_t$$, and ii) removes the need to solve an expensive linear system of size $$d$$.
Hence, quasi-Newton methods require only quadratic cost, both in memory and compute (the latter to perform the matrix vector multiplication with the gradient)
in contrast to quadratic and cubic cost respectively of Newton's method.

### Limited-memory BFGS (L-BFGS) scales linearly

Quasi-Newton methods can be made even faster such that they scale linearly with $$d$$ in memory and compute. 

To see this, first observe that the quasi-Newton step
$$p_t^{\mathrm{quasi}}$$ can be constructed in two different ways.
The straightforward way is to keep the matrix $$B_{t}^{-1}$$ in memory and to update it at every iteration with the 
rank-2 term as in Eq. \eqref{eq:dennis} using the newly acquired gradient $$\nabla f(w_{t})$$ and path segment $$s_{t-1}$$. 
Then $$B_t^{-1}$$ can be multiplied with the
current gradient to obtain $$p_t^{\mathrm{quasi}}$$.

An alternative way is to never construct nor store $$B_t^{-1}$$ directly, but to keep all 
paths segments $$\{s_i\}_{i=0}^{t-1}$$ and gradients $$\{\nabla f(w_i)\}_{i=0}^{t}$$ in memory instead.
The step $$p_t^{\mathrm{quasi}}$$ is then constructed directly by adding up the analytic products of the 
$$t$$ rank-2 terms with their corresponding gradients. 

This second version thus requires storing $$2t-1$$ $$d$$-dimensional vectors (the path segments and the gradients) and requires
$$\mathcal{O}(td)$$ operations. Hence, if $$t\ll d$$, this version of computing the quasi-Newton update is linear in $$d$$.

#### A shifting window of memory
The simple trick of *limited memory* quasi-Newton methods such as *limited memory BFGS* (L-BFGS)
is now to simply only keep a shifting window of the last $$m$$ gradients and corresponding path segments in memory instead of the whole history.
This reduces the memory cost such that only $$2m-1$$ $$d$$-dimensional vectors need to be stored; the compute cost is reduced to $$\mathcal{O}(d)$$.

In practice, any value between $$m=1,...20$$ is generally used. 
In my experience often even $$m=1$$ which equals to keeping only one past gradient and path segment
in memory and which yields a rank-2 approximation performs very well. 

In my experience, limited memory BFGS often even outperforms its original 'more precise' version with an unlimited memory.
At first, this may sound counterintuitive as L-BFGS uses less observed gradients to build an approximation to the Hessian.
But keep in mind, that the estimator $$B_t$$ can be interpreted as the sum of rank-2 matrices each of which encodes information about the 
average Hessian along their respective path segment. Thus, in case the true Hessian function $$\nabla^2 f(w)$$ changes 
considerably along the optimizer's path, it may even be suboptimal to keep around old and potentially outdated rank-2 terms that have nothing to do 
anymore with the Hessian at the current location $$w_t$$. Newton's method never has this problem, as the Newton step is computed with
local information at $$w_t$$ only---it does not have a memory, and each Newton step is independent of the previous one.

The precise algorithm for L-BFGS can for example be found [here](https://en.wikipedia.org/wiki/Limited-memory_BFGS).

### Line searches: The key to success

We've left out a huge, and unfortunately often neglected component of quasi-Newton methods: Line searches!

In essence, line searches scale the quasi-Newton step with a positive scalar $$0<\alpha_t<\infty$$
such that the update becomes 

$$
w_{t+1} = w_t + \alpha_tp_t^{\mathrm{quasi}}
$$

While line searches play a minor role in Newton's method and are mainly used to stabilize
its initial iterates, they play an *integral part* in quasi-Newton methods and cannot be separated from them.

Spoiler: Line searches ensure (for some quasi-Newton methods such as BFGS) that the estimator $$B_t$$
(and hence $$B_t^{-1}$$) is always positive definite even if the underlying true Hessian $$\nabla^2 f(w_t)$$ is not!
Thus, quasi-Newton methods can ensure a descent direction, and can be applied to non-convex objective function $$f$$
where Newton's method fails.

As this post is getting a bit long, I may discuss line searches and their role in Newton's and quasi-Newton methods in another post, hopefully to come soon.
[Edit: I actually made one. It's [here]({% link _posts/2023-02-19-line-searches.md %})]

## Lastly (there is much more)

There is so much more to say about Newton's and quasi-Newton methods which we haven't touched on here such as
convergence rates etc. Further, there are intricate connections to linear solvers such as 
the *conjugate gradient algorithm* (CG) which coincides with BFGS in some sense under certain circumstance.
I suppose, whenever a linear system is solved explicitely (as in Newton's method) or implicitly (as in quasi-Newton methods)
you have to think about linear solver, too. But this, too, may be a topic for another blogpost.

## References

[1] J. J. Dennis and J. Moré 1977 *Quasi-Newton methods, motivation and theory.*, SIAM Review 19.1 pp. 46–89.

[2] J. Dennis 1971 *On some methods based on Broyden’s secant approximations.*, Numerical Methods for Non-Linear Optimization.

[3] C. Broyden 1969 *A new double-rank minimization algorithm*, Notices of the AMS 16.4 p. 670.

[4] R. Fletcher 1970 *A new approach to variable metric algorithms*, The Computer Journal 13.3 p. 317.

[5] D. Goldfarb 1970 *A family of variable metric updates derived by variational means*, Math. Comp. 24.109 pp. 23–26.

[6] D. Shanno 1970 *Conditioning of quasi-Newton methods for function minimization*, Math. Comp. 24.111 pp. 647–656.

[7] J. Greenstadt 1970 *Variations on variable-metric methods*, Math. Comp 24 pp. 1–22.

[8] M. Powell 1970 *A new algorithm for unconstrained optimization*, Nonlinear Programming. 

[9] W. Davidon 1959 *Variable metric method for minimization*, Tech. rep. Argonne National Laboratories, Ill.

[10] R. Fletcher and M. Powell 1963 *A rapidly convergent descent method for minimization* The Computer Journal 6.2 pp. 163–168.
