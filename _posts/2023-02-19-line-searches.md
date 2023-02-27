---
layout:     post
title:      "Line Searches"
author:     mmahsereci
description:    ""
date:       2023-02-19
category:   techblog
tags:       [optimization, machinelearning]
description: >
    Line searches are fast an efficient sub-routines that determine
    the step size (a.k.a 'learning rate') of gradient-based optimizers at every iteration.
    Besides this, line searches have auxiliary purpose in quasi-Newton methods, where a correctly
    chosen step size yields positive definite Hessian estimates and thus descent directions.
    In this post, we discuss two well-known instances of a line search and their use cases: 
    1) the back-tracking line search, and 2) line searches based on cubic polynomials and the Wolfe conditions.

authors:
  - name: mmahsereci
    url: "https://github.com/mmahsereci"
    affiliations:
      name: University of T&uuml;bingen
---

Line searches are fast an efficient sub-routines that determine
the step size (a.k.a 'learning rate') of gradient-based optimizers at every iteration.
Besides this, line searches have auxiliary purpose in quasi-Newton methods, where a correctly
chosen step size yields positive definite Hessian estimates and thus descent directions.
In this post, we discuss two well-known instances of a line search and their use cases: 
1) the back-tracking line search, and 2) line searches based on cubic polynomials and the Wolfe conditions.

## Optimization objective and notation

Please refer to [this]({% link _posts/2023-01-15-quasi-newton-methods.md %}) post for notation on the 
optimization objective $$f: \mathbb{R}^d\rightarrow \mathbb{R}$$ and its gradient and
Hessian function $$\nabla f: \mathbb{R}^d\rightarrow \mathbb{R}^d$$ and $$f: \mathbb{R}^d\rightarrow \mathbb{R}^{d\times d}$$ respectively. 
We re-iterate that we aim to solve the optimization problem 

$$ 
w^∗ = \operatorname*{arg\,min}_w f (w).
$$

Further, whenever the Newton step, the quasi-Newton step, the Hessian, or the quasi-Newton estimate $$B_t$$ of the Hessian
is mentioned, please also refer to the [mentioned post]({% link _posts/2023-01-15-quasi-newton-methods.md %}) 
for notation (this is mainly needed for the section on the role of line searches in quasi-Newton methods). 

## What do line searches do?

Let $$p_t\in\mathbb{R}^d$$ be any descent direction and 

$$
w_{t+1} = w_t + p_t
$$

a typical update of an optimizer. 

Line searches are routines that are called at every iteration $$t$$ *after* the descent direction $$p_t$$ is determined
by the optimizer and *before* $$w_{t+1}$$ is computed. 

Their task is to find a suitable scalar $$0< \alpha_t < \infty$$ that yields a *scaled step* $$\alpha_tp_t$$. 
The update then becomes

$$
w_{t+1} = w_t + \alpha_tp_t.
$$

The scaling of the step is supposed to stabilize the optimizer and increase its efficiency.

### Line searches operate on 1-dimensional problems

As mentioned, line searches come into play after $$p_t$$ is determined.
Hence, line searches only ever operate on a 1-dimensional problem defined as 

$$
\begin{alignat*}{2}
\alpha & \geq 0\qquad &&\text{(1-dim domain)}\\
w(\alpha) & := w_t + \alpha p_t\qquad &&\text{(1-dim subspace)}\\
h(\alpha) & := f(w(\alpha)). &&\text{(1-dim objective)}
\end{alignat*}
$$

Putting the above formulas into words: The line search objective is $$h: \mathbb{R}_{+}\rightarrow \mathbb{R}$$, $$\alpha\mapsto h(\alpha)$$ 
and has domain $$\mathbb{R}_+$$. In terms of $$w$$, the domain is equal to the 1-dimensional slice $$\{w(\alpha) \vert \alpha\in\mathbb{R}_+\}$$ 
through the $$d$$-dimensional $$w$$-space 'to-the-right' (for positive $$\alpha$$) of $$w_t$$.
The function values of $$h$$ are simply the corresponding function values of $$f$$ on that line. 

Further, the corresponding 1-dimensional gradient $$h': \mathbb{R}_{+}\rightarrow \mathbb{R}$$, $$\alpha\mapsto h'(\alpha)$$ is

$$
\begin{align*}
h'(\alpha) &= (\frac{\partial w}{\partial \alpha})^{\intercal}\nabla f(w(\alpha))\\
 &= p_t^{\intercal}\nabla f(w(\alpha)),
\end{align*}
$$

which is equal to a projection of the high-dimensional gradient $$\nabla f(w(\alpha))$$ onto $$p_t$$.

All computations inside the line search routine exclusively use the univariate quantities 
$$\alpha$$, $$h(\alpha)$$ and $$h'(\alpha)$$. Thus the line search routine itself scales $$\mathcal{O}(1)$$
w.r.t the problem dimension $$d$$.

## What is a good step size?

We learned that line searches automatically find suitable step sizes $$\alpha_t$$ at every iteration $$t$$ of the optimizer.
But what exactly does 'suitable' mean? 
There are some general rules, although the type of the optimizer that produces $$p_t$$ will mostly define this. 
Consequently, certain instances of line searches are typically used alongside certain optimizers
(textbooks like [Nocedal & Wright](#references) give a good overview).

Let us have a look at an example: The backtracking line search used with the Newton optimizer.

### Example 1: Backtracking line searches for Newton's method

The Newton optimizer updates $$w_t$$ according to the Newton step $$p_t^{\mathrm{newton}}$$.
Recall from the [previous post]({% link _posts/2023-01-15-quasi-newton-methods.md %})  
that the Newton step is given by  $$p_t^{\mathrm{newton}}:=-\nabla^2 f(w_t)^{-1} \nabla f(w_t)$$
where $$\nabla^2 f(w_t)^{-1}$$ is the inverse Hessian at location $$w_t$$. 
Let us assume here for simplicity that the Hessian at $$w_t$$ is positive definite and 
that the Newton step yields a descent direction.
The step $$p_t^{\mathrm{newton}}$$ is equal to stepping into the minimizer of a local quadratic approximation of $$f$$ according 
to a second order Taylor expansion around $$w_t$$.

From what we just learned, we can deduce that the Newton step has a natural scale of $$\alpha=1$$ 
that steps into said minimizer of the Taylor expansion.
Sometimes though, when the Taylor expansion is not accurate enough, the step may overshoot the true objective, 
especially at the beginning of the optimization when $$w_t$$ is still too far away from the minimizer of $$f$$, causing the optimizer to diverge. 
Hence, Newton's method is often used with a *backtracking line search*.

A backtracking line search starts with the natural guess $$\alpha = 1$$ and then shortens the step e.g., by a constant 
multiplicative factor. The line search terminates once a suitable step is found, or throws an error instead.
Simple code may look for example like this:

```python
def backtracking_ls(h, h0):
    """h0 is the function value at the current position (alpha=0)"""
    alpha = 1  # initial guess for the step size
    gamma = 0.9  # multiplicative factor 
    n_evals_max = 20  # maximal number of evaluations per line search
    n_evals = 0  # number of evaluations performed during the line search
    while n_evals < n_evals_max:
        halpha = h(alpha)  # evaluate h at current guess
        n_evals += 1
        # is the step suitable?
        if stopping_condition(h0, halpha):
            return alpha
        alpha = gamma * alpha  # reduce the step size
    raise RuntimeError("A suitable step size could not be found.")
```

We observe that the line search probes the natural $$\alpha=1$$ first, and returns if this initial guess is suitable.
Hence, a line search can terminate with no additional function evaluations in comparison to 
an optimizer that uses no line search.

If the guess is not suitable, the step is shortened by a multiplicative factor $$\gamma\in(0, 1)$$. 
If no suitable step can be found in the given budget of ``n_evals_max`` function evaluations, the 
line search raises an exception.

We summarize that the backtracking line search only intervenes (shortens the step) and uses additional compute if the 
initial step size would most likely derail the optimizer. This is an important property since
line searches are supposed to be lightweight. 

#### The Armijo condition for backtracking line searches

A backtracking line search usually uses the [Armijo condition](#references) as ``stopping_condition(...)``
which encodes that the function value $$f(w_{t+1})$$ should decrease linearly 
along the line compared to the current value $$f(w_t)$$. 
Thus, the Armijo condition is also called the *sufficient decrease* condition; it reads:

$$
\begin{alignat*}{2}
h(\alpha) \leq h(0) + \alpha c_1 h'(0)\quad &&\text{(Armijo condition)}
\end{alignat*}
$$

where $$c_1\in[0, 1)$$ is a fixed constant usually close or equal to $$0$$ (demanding a trivial decrease of the objective function;
note that $$h'(0)$$ is negative if $$p_t$$ is a descent direction which is generally assumed for line searches).

Newton's optimizer using a backtracking line search may look something like this:

```python
import numpy as np

def _h(alpha, f, w, p):
    """helper function to construct h."""
    fw, dfw, _ = f(w + alpha * p)
    return fw, np.dot(dfw, p)

T = 1000  # maximum number of optimization steps
w = w0  # w0 is the starting position of the optimizer
for _ in range(T):
    fw, dfw, ddfw = f(w)  # evaluate objective, dfw and ddfw are gradient and Hessian at w
    p = get_newton_direction(dfw, ddfw)
    alpha = backtracking_ls(h=lambda a: _h(a, f, w, p), h0=fw)
    w += alpha * p  # step
```

(Note: The code snippets above are inefficient implementations because $$f$$ might be
evaluated twice at the accepted step (once inside the line search when evaluating $$h$$ 
and once explicitly in the main optimizer loop). This compute-sharing is easy to resolve, but it would 
convolute the code which is why I opted for the shown version.)

### Example 2: Line searches based on Wolfe-conditions for quasi-Newton methods

The backtracking line search is a rather simple line search. For instance the search space for
$$\alpha$$ is restricted as it is capped by a maximal value of $$\alpha=1$$, and we were only
interested in *shortening* a step, hence the Armijo condition was appropriate 
(the Armijo condition in principle allows very small, inefficient steps, but as we were backtracking this was not an issue).
This strategy is suitable for Newton's method as it exhibits a natural scale that is often right.

When used with other optimizers, line searches often cannot be as simple and need to allow for any continuous $$\alpha\in\mathbb{R}_+$$,
and be able to both extend and shorted the step.
Thus, line searches often do the heavy lifting when it comes to stabilizing optimizers.

In particular, *line searches base on the Wolfe conditions* are essential to stabilize e.g., the (L-)BFGS
update and even ensure a descent direction for non-convex $$f$$. 
(At this point it may make sense to read the [previous post]({% link _posts/2023-01-15-quasi-newton-methods.md %}) 
on quasi-Newton methods as I will not explain their update here again.
We will simply note that the quasi-Newton update $$p_t^{\mathrm{quasi}} = - B_t^{-1}\nabla f(w_t)$$ approximates the Newton step
using $$B_t^{-1}$$, an evolving estimate of the inverse Hessian constructed from past gradient difference 
$$\Delta y_{t-1}:= \nabla f(w_{t}) - \nabla f(w_{t-1})$$ 
and path segment observations $$s_{t-1} := w_{t} - w_{t-1}$$.)

The [Wolfe conditions](#references) are often motivated as an extension of the Armijo condition, that, in addition to
preventing too large steps (by capping the allowed function value), also prevent too short steps
(by imposing an increase in gradient).
However, we will motivate the Wolfe conditions a bit non-traditionally here by examining the BFGS update. 

#### The Wolfe conditions (Line searches can ensure positive definite BFGS updates)

As shown in the [previous post]({% link _posts/2023-01-15-quasi-newton-methods.md %}) (L-)BFGS 
does not explicitly encode positive definiteness of the Hessian estimate $$B_t$$. 

However, Theorems 7.7 and 7.8 in [Dennis and Moré](#references) show (by examining the determinant) 
that $$B_t$$ of BFGS (and DFP for that matter)
are positive definite if the newly collected gradient difference and path segment fulfill
$$s_t^{\intercal}\Delta y_t > 0$$. Hence, let us define the parametrized path segment
$$s(\alpha) := w(\alpha) - w_t = \alpha p_t$$ and the parametrized gradient difference 
$$\Delta y(\alpha):=\nabla f(w(\alpha)) - \nabla f(w(0))$$
and let us write

$$
\begin{align*}
s(\alpha)^{\intercal}\Delta y(\alpha) & = \alpha p_t^{\intercal}\nabla f(w(\alpha)) - \alpha p_t^{\intercal}\nabla f(w(0))\\
& = \alpha h'(\alpha) - \alpha h'(0) > 0\\
\rightarrow h'(\alpha) & > h'(0).
\end{align*}
$$

The above result states that $$B_t$$ will stay positive definite if $$h'(\alpha)  > h'(0)$$.
That is, a suitable step $$\alpha$$ must have a larger projected gradient
$$h'(\alpha)$$ in comparison to the current projected gradient $$h'(0)$$.
Since $$h'(0)$$ is negative (descent direction), this means that $$h'(\alpha)$$ must either be
smaller in absolute value or positive.
Hence, when (L-)BFGS is used with a line search that finds step sizes that fulfill the above inequality, the estimator $$B_t$$ 
will always be positive definite, and hence $$p_t$$ will always be a descent direction.
**This holds even if $$f$$ is non-convex and the true Hessian matrix at $$w_t + s(\alpha)$$ is not positive definite.**

The condition we just found is called the *curvature condition*, and together with the Armijo condition
forms the *weak Wolfe conditions*

$$
\begin{alignat*}{2}
h(\alpha) &\leq h(0) + \alpha c_1 h'(0)\quad &&\text{(Wolfe-I, or Armijo condition)}\\
h'(\alpha) &> c_2 h(0)\quad &&\text{(Wolfe-II, or curvature condition)}
\end{alignat*}
$$

where $$0<c_1<c_2<1$$. As mentioned above, the curvature condition can also be derived independently of the
definiteness constraint for the BFGS update, as a means to disallow short steps.

Still, let us stop here for a moment and appreciate the interplay between the seemingly separate BFGS quasi-Newton estimate $$B_t$$
and the line search routine enforcing the Wolfe conditions:
The line search can affect the definiteness of the BFGS update by choosing the step size as illustrated above.
Furthermore, the BFGS update rule allows ($$B_t$$ remains positive definite) for just the right size of steps for the optimizer to make rapid progress.
The latter should be understood in the sense that the Wolfe conditions are meaningful in their own right and have been derived in other contexts, 
for example as conditions for convergence. 
To me, this interplay between two a priori separate methods with separate concerns is a true masterpiece of algorithm, and 
it may explain in parts the huge success of the (L-)BFGS method over several decades to date.

(A note in passing: It is not possible to control the robustness of Newton's method in the same way.
This is because the Newton step uses the Hessian matrix at location $$w_t$$ which simply is positive definite, or it is not.)

Continuing on, in practice, often the *strong Wolfe conditions* are used over the weak ones, as they safeguard a bit better against too large steps.
The strong Wolfe conditions modify the curvature condition and disallow too large positive gradients. In essence, the projected gradient must 
not only become larger, but also decrease in absolute value under the strong Wolfe conditions.

$$
\begin{alignat*}{2}
h(\alpha) &\leq h(0) + \alpha c_1 h'(0)\quad &&\text{(Wolfe-I, or Armijo condition)}\\
|h'(\alpha)| & < c_2 |h(0)|\quad &&\text{(Wolfe-IIa, or curvature condition)}
\end{alignat*}
$$

(Note that the sign changed from a larger-than to a smaller-than sign in the second line due to taking the
absolute values.)
The weak Wolfe conditions contain the strong Wolfe conditions in the sense that all $$\alpha$$ allowed by
the strong, will also be allowed by the weak Wolfe conditions.


#### Polynomial interpolants (or how to search the line)

The Wolfe conditions define a suitable step; now let us see how to efficiently
search the line. In contrast to the backtracking line search that only searches on the domain $$\alpha=(0, 1]$$
we now want to allow for $$\alpha\in\mathbb{R}_+$$ since Wolfe points may lie at larger values $$\alpha>1$$, too.

We cannot simply sample a whole range of points, as this would decrease the efficiency of the line search significantly.
Thus, line searches often try the step size of the previous iteration first, and, if not accepted, extend the step
with a multiplicative factor until a point with positive gradient $$h'(\alpha)>0$$ is reached 
(this means there must exist a minimizer between this candidate and the previous one tried).
Let $$\alpha^{(i)}$$, $$i=0, 1, 2, \dots$$ be the guesses. 
The extension phase is followed by a search in $$[0, \alpha^{(i)}]$$ or $$[\alpha^{(i-1)} , \alpha^{(i)}]$$, 
by interval nesting or by interpolation of the collected function and gradient values,
e.g. with [cubic splines](https://en.wikipedia.org/wiki/Spline_(mathematics)). 
If cubic splines are used, the next candidate may be the minimizer of the spline.
If any point during the described procedure is a Wolfe point (also the initial try), the line search terminates.

The described procedure may sound complicated (see e.g., [Nocedal & Wright](#references) for an extended discussion), 
but line searches of this sort are highly efficient in probing 
the right candidates $$\alpha$$ such that a Wolfe point is found immediately or with very little 
function evaluations. To make things a bit clearer, here is a sketch of a potential line search using the Wolfe conditions:

```python
def wolfe_ls(h, h0, dh0, alpha):
    """alpha is the step size of the previous iteration."""
    # set up storage (all stored values are scalars)
    Alphas = [0]  # h0 and dh0 correspond to alpha=0
    H = [h0]
    dH = [dh0]
    
    n_evals_max = 20  # maximal number of evaluations per line search
    n_evals = 0  # number of evaluations performed during the line search    
    extrapolate = True  # flag for extrapolation phase
    while n_evals < n_evals_max:
        halpha, dhalpha = h(alpha)  # evaluate h at current guess
        n_evals += 1
        
        # update storage
        Alphas.append(alpha)
        H.append(halpha)
        dH.append(dhalpha)

        # is the step suitable?
        if wolfe_conditions(h0, dh0, halpha, dhalpha, alpha):
            return alpha
        alpha, extrapolate = compute_next_candidate(alpha, Alphas, H, dH, extrapolate)
    raise RuntimeError("A suitable step size could not be found.")

def compute_next_candidate(alpha, Alphas, H, dH, extrapolate):
    # stop extrapolating if gradient turns positive first time
    if extrapolate and dH[-1] > 0:
        extrapolate = False

    if extrapolate:
        return alpha * 2, extrapolate  # double step size

    # find e.g., cubic minimizer of previous cell
    return cubic_minimizer(alpha, Alphas, H, dH), extrapolate
```

In summary, more involved line searches operate on a larger, possibly unbounded search space for $$\alpha$$,
they use more complex stopping criteria, such as the Wolfe conditions, and they can both extend and shorted
the steps of the optimizer. Apart from step size, they can control properties of some selected optimizers
as could be seen from the BFGS update where the line search ensures future descent directions $$p_{t+1}$$.

## Some final thoughts

Line searches are, in my humble opinion, really cool subroutines that get glossed over often.
From a practical perspective line searches automate one of the most sensitive 
hyperparameter of optimization: The learning rate, which is extremely tedious if it
needed to be set by hand. Hence, line searches might seem like small/ non-essential subroutines at first glance, 
but often they are at the heart of optimizers that solve the most intricate problems automated and efficiently.
Robust implementations go beyond the code snippets shown above, as one main objective of a line
search is to 'keep the optimizer going' also in numerically unstable situation, and even when 
no suitable point can be found in the given budget. 

You may finally be wondering: But what about deep learning? The short answer is that things are simply
different if the objective $$f$$ cannot be evaluated precisely (due to mini-batching). In this case
many assumptions of the line search go down the drain (a straightforward example is that the Wolfe conditions cannot be checked with certainty anymore;
an even more straightforward example is that descent directions $$p_t$$ cannot be ensured anymore).
I may do another blogpost at some point about probabilistic equivalents of line searches that are robust to gradient noise and
that perform really well for certain types of deep learning problems, but alas not for all of them yet.


## References

[1] J. J. Dennis and J. Moré 1977 *Quasi-Newton methods, motivation and theory*, SIAM Review 19.1 pp. 46–89.

[2] L. Armijo 1966 *Minimization of functions having Lipschitz continuous first partial derivatives*, Pacific Journal of Mathematics 16.1 pp. 1–3.

[3] P. Wolfe 1969 *Convergence conditions for ascent methods*,  SIAM Review 11.2 pp. 226–235.

[4] J. Nocedal, S. J. Wright 2006 *Numerical Optimization*, Springer.

[5] (some of the above text is copied from my PhD thesis, Section 2.5) M. Mahsereci 2018 *Probabilistic Approaches to Stochastic Optimization*.

## Appendix: A practical note on line search parameters

We accumulated a bunch of line search parameters ($$\gamma$$, $$c_1$$, $$c_2$$, ...) that 
(at least if you grew up with deep learning, which this post is not about) seem like an extra nuisance to tune,
defeating the purpose of any kind of automation. But fortuitously, in the deterministic world, it can be 
said that those parameters can be fixed and stay fixed for pretty much any problem without compromising the
optimization performance. The reasons are that i) the optimizer's effectiveness does not seem to 
depend on the parameter values a lot, and ii) line searches can be quite lenient, allowing for most steps (e.g., $$c_1$$ close to 0)
and only disallowing catastrophic steps in order to keep the optimizer doing its thing. The latter can be understood in the
context of a line search being an auxiliary step within a larger iteration, and thus the definition of a 'suitable step' does not need to be very precise. 

*Small caveat:* There is usually no need at all to meddle with the parameters of the Wolfe or the Armijo condition, but it may 
be helpful here and there to play with $$\gamma$$ of a backtracking line search or the backtracking schedule
when evaluating $$f(w)$$ is very expensive by itself *and* expensive in comparison to computing $$p_t$$.
In this case one would rather shorten the step quite aggressively (smaller $$\gamma$$) and find "some step" for a reduced budget that trying
to find a more precise but costly step. 
The parameter $$\gamma$$ or the schedule can be fixed by some meaningful expert knowledge 
(how many function evaluations do I want to waste in the worst case and what's the shortest step I want to try?) rather than by trial and error.
If $$\gamma$$ is used, the step sizes tried are equal to $$\gamma^i$$ for $$i=0, 1, 2, \dots$$ which may give a hint on what to do best.
