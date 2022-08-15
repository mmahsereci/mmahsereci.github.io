---
layout:     post
title:      "The Kronecker Product"
author:     mmahsereci
description:    ""
date:       2022-08-05
category:   techblog
tags:       [machinelearning]
description: >
    The Kronecker product is a tensor product that appears frequently in machine learning
    and its applications.
    This post collects some useful properties and some other insights on it.

authors:
  - name: mmahsereci
    url: "https://github.com/mmahsereci"
    affiliations:
      name: University of T&uuml;bingen
---
  
The Kronecker product is a tensor product that appears frequently in machine learning
and its [applications](#applications-of-the-kronecker-product).
This post collects some useful properties and some other insights on it.

> There is a follow-up post on the lesser known symmetric and anti-symmetric 
> Kronecker product [here]({% link _posts/2022-08-06-kronecker-sym.md %}).

Let $$A\in\mathbb{R}^{n_1\times n_2}$$ and $$B\in\mathbb{R}^{m_1\times m_2}$$ be two matrices. 
Then, the Kronecker product, denoted `$$\otimes$$', of $$A$$ and $$B$$ is in $$\mathbb{R}^{n_1m_1\times n_2m_2}$$ and defined as

$$
\begin{equation}
\label{eq:def}
\begin{split}
(A\otimes B)_{(ij), (kl)} = A_{ik} B_{jl},\quad 
&i=1\dots n_1,~ k=1\dots n_2\\
&j=1\dots m_1,~ l=1\dots m_2,
\end{split}
\end{equation}
$$

where $$(ij)$$ and $$(kl)$$ are the double indices 
$$(ij) = m_1 (i-1) + j$$ and $$(kl)=m_2(k-1) + l$$.

An example for $$n_1=n_2=m_2=2$$ and $$m_1=3$$ is

$$
\left[\begin{array}{c c} 
A_{11} & A_{12} \\
A_{21} & A_{22} 
\end{array}\right]
\otimes
\left[\begin{array}{c c} 
B_{11} & B_{12} \\
B_{21} & B_{22} \\
B_{31} & B_{32}
\end{array}\right]
=
\left[\begin{array}{c c | c c} 
A_{11}B_{11} & A_{11}B_{12} & A_{12}B_{11} & A_{12}B_{12}\\
A_{11}B_{21} & A_{11}B_{22} & A_{12}B_{21} & A_{12}B_{22} \\
A_{11}B_{31} & A_{11}B_{32} & A_{12}B_{31} & A_{12}B_{32} \\
\hline
A_{21}B_{11} & A_{21}B_{12} & A_{22}B_{11} & A_{22}B_{12}\\
A_{21}B_{21} & A_{21}B_{22} & A_{22}B_{21} & A_{22}B_{22} \\
A_{21}B_{31} & A_{21}B_{32} & A_{22}B_{31} & A_{22}B_{32} \\
\end{array}\right]
$$.

We observe that the Kronecker product has block structure

$$
A\otimes B = 
\left[\begin{array}{c | c | c | c} 
A_{11} B & A_{12} B &\cdots & A_{1n_2} B\\
\hline
A_{21} B & A_{22} B &\cdots & A_{2n_2} B \\
\hline
\vdots & \vdots &\ddots  & \vdots\\
\hline
A_{n_1 1} B & A_{n_1 2} B &\cdots  & A_{n_1n_2} B
\end{array}\right]
$$

with $$n_1\times n_2$$ blocks where each block is of size $$m_1\times m_2$$ and contains 
the matrix $$B$$ multiplied with one of the elements of $$A$$. This means, a block-diagonal 
matrix with $$n$$ identical blocks $$B$$ can be written as $$I\otimes B$$, where $$I\in\mathbb{R}^{n\times n}$$
is the identity matrix.


## Kronecker algebra

The Kronecker product has some nice algebraic properties which roughly resemble the ones 
of rank-one matrices. For matrices $$A$$, $$B$$, $$C$$ and $$D$$ of appropriate sizes 
and properties it is

$$
\begin{alignat}{2}
\label{eq:1}
&\text{transpose}           &(A\otimes B)^{\intercal} &= (A^{\intercal}\otimes B^{\intercal}) \\
\label{eq:2}
&\text{inverse}             &(A\otimes B)^{-1} &= A^{-1}\otimes B^{-1} \\
&\text{factorizing}         &(A\otimes B)(C\otimes D)&= (AC\otimes BD) \\
&\text{distributive left}\qquad\qquad   &(A\otimes B) + (A\otimes C) &= A\otimes (B + C) \\
&\text{distributive right}  &(A\otimes B) + (C\otimes B)&= (A+C)\otimes B\\
\label{eq:6}
&\text{associative}         &(A\otimes B)\otimes C&=  A\otimes (B\otimes C)
\end{alignat}
$$

where $$A^{\intercal}$$ denotes the transpose and $$A^{-1}$$ the inverse (should it exist) 
of the matrix $$A$$, and respectively for the other matrices. 
Proof of all these equalities follow straightforwardly from the definition of the 
Kronecker product (Eq. \eqref{eq:def}) but can be found in the [references](#references) below.

Eqs. \eqref{eq:1}-\eqref{eq:6} especially hold if $$A$$, $$B$$, $$C$$, $$B$$ are scalars or 
vectors (where applicable). 

All formulas exploit the factorizing structure of the Kronecker product, which is why one side 
of the equality may be much cheaper to compute than the other. 
For example, the left-hand side of Eq. \eqref{eq:2}, 
requires the inverse of a full $$nm\times nm$$ matrix which is $$\mathcal{O}(n^3m^3)$$, 
while the right-hand side only requires the inverse of an $$n\times n$$ and $$m\times m$$ matrix
which is $$\mathcal{O}(n^3 + m^3)$$.
Properties like this make the Kronecker product interesting for large scale applications.

In general, the Kronecker product does not commute

$$
\begin{equation*}
A\otimes B \neq B\otimes A.
\end{equation*}
$$

The rank of $$A\otimes B$$ is the product of the ranks of $$A$$ and $$B$$ 

$$
\operatorname{rk}[A \otimes B ] = \operatorname{rk}[A] \operatorname{rk}[B].
$$

Specifically, if $$A$$ and $$B$$ are square and have full rank, $$A\otimes B$$
is square and has full rank, too.

For square matrices $$A\in\mathbb{R}^{n\times n}$$ and $$B\in\mathbb{R}^{m\times m}$$,
the determinant of their Kronecker product is given by the product of powers of the
individual determinants

$$
\det(A\otimes B) = \det(A)^{m}\det(B)^n.
$$

The trace of the Kronecker product is the product of the individual traces

$$
\operatorname{tr}[(A\otimes B)]= \operatorname{tr}[A]\operatorname{tr}[B].
$$

## Vector multiplication

Define the vectorization operation 

$$
\overrightarrow{\phantom{X}}:\mathbb{R}^{a\times b}\to \mathbb{R}^{ab},\quad
X\mapsto \overrightarrow{X}
$$ 

as stacking the rows of a matrix into a vector. 
A consequence of Eq. \eqref{eq:def} is that a Kronecker product applied to a vectorized matrix 
$$\overrightarrow{X}$$ of appropriate size is the vectorized version of two lower-dimensional, 
cheaper matrix-matrix multiplications

$$
\begin{equation}
\label{eq:vec}
(A \otimes B ) \overrightarrow{X} = \overrightarrow{AXB^{\intercal}}.
\end{equation}
$$

Eq. \eqref{eq:vec} follows directly from Eq. \eqref{eq:def} and the definition of the 
vectorization operation.
Again, due to the factorization property, the right-hand side amounts to 
multiplying two smaller matrices while the left-hand side amounts to a large matrix-vector
multiplication.

An alternative definition of the vectorization operation is 
stacking the columns of a matrix. We then obtain a similar formula 
$$(A \otimes B ) \overrightarrow{X} = \overrightarrow{BXA^{\intercal}}$$ where the 
locations of $$A$$ and $$B$$ are swapped on the right-hand side. 
For the remained of this blog post, we will stick with Eq. \eqref{eq:vec} though. 


## Relation to the Frobenius norm

Let $$A\in\mathbb{R}^{n_1\times n_2}$$ be a matrix. 
The Frobenius norm $$\|\cdot\|_F: \mathbb{R}^{n_1\times n_2}\to \mathbb{R}_{0,+}$$ 
is a matrix norm defined as

$$
\begin{equation}
\begin{split}
\|A\|_F^2 
= \operatorname{tr}[A^{\intercal}A]
= \sum_{i=1}^{n_1} \sum_{j=1}^{n_2}A_{ij}^2
&= \|\overrightarrow{A}\|^2\\
&= \overrightarrow{A}^{\intercal}\overrightarrow{A}\\
&= \overrightarrow{A}^{\intercal}(I_{n_1}\otimes I_{n_2})\overrightarrow{A},
\end{split}
\end{equation}
$$

where $$I_{n_1}$$ and $$I_{n_2}$$ are identity matrices of sizes $$n_1$$ and $$n_2$$
respectively and $$\|\cdot\|$$ is the Euclidean norm.

Let $$W\in\mathbb{R}^{n\times n}$$ be a positive definite matrix and 
$$A\in\mathbb{R}^{n\times n}$$ be an arbitrary square matrix.
The weighted Frobenius norm is defined as

$$
\begin{equation}
\begin{split}
\|A\|_{F, W}^2 
= \|W^{\frac{1}{2}}AW^{\frac{1}{2}}\|_F^2
&= \operatorname{tr}[WA^{\intercal}WA]\\
&= \sum_{i, j, k, l=1}^{n}A_{ji}W_{jk}W_{il}A_{kl}\\
&= \overrightarrow{A}^{\intercal}(W\otimes W)\overrightarrow{A}.
\end{split}
\end{equation}
$$

Hence, the weighted square Frobenius norm can be expressed as an inner product weighted
with a positive definite Kronecker matrix.

We can generalize the weighted Frobenius norm to allow two positive definite weight matrices
$$W_1\in\mathbb{R}^{n_1\times n_1}$$ and $$W_2\in\mathbb{R}^{n_2\times n_2}$$
and a non-square $$A\in\mathbb{R}^{n_1\times n_2}$$ such that

$$
\begin{equation*}
\begin{split}
\|A\|_{F, W_1, W_2}^2 
&= \operatorname{tr}[W_2A^{\intercal}W_1A]\\
&= \overrightarrow{A}^{\intercal}(W_1\otimes W_2)\overrightarrow{A}.
\end{split}
\end{equation*}
$$

It is mentioned here since used in the [applicattion section](#applications-of-the-kronecker-product).

## Closest Kronecker product

The solution to the closest Kronecker approximation problem gives some insight into
the factorization structure of the Kronecker product which is why it is mentioned
here specifically. 

Suppose we are given a large matrix $$C\in\mathbb{R}^{n_1m_1\times n_2m_2}$$. 
The closest Kronecker product $$A^*\otimes B^*$$ under the Frobenius norm is given by

$$
\begin{equation}
\label{eq:argmin}
A^*, B^* = \operatorname*{arg\,min}_{A, B}\|C-A\otimes B\|_F^2.
\end{equation}
$$

There exists a fixed, known permutation 
$$\mathcal{R}: \mathbb{R}^{n_1m_1\times n_2m_2}\to \mathbb{R}^{n_1n_2\times m_1m_2}$$
that vectorizes and stacks blocks of $$C$$
([[1]](#references) Section 6) such that the Kronecker
product can be written as an outer product of the vectorized matrices $$\overrightarrow{A}$$
and $$\overrightarrow{B}$$ according to 
$$\mathcal{R}(A\otimes B) = \overrightarrow{A}\overrightarrow{B^{\intercal}}$$ .
Thus, Eq. \eqref{eq:argmin} can be re-phrased as a rank-one approximation problem in an 
$$n_1n_2\times m_1m_2$$ dimensional space

$$
\begin{equation*}
\label{eq:argmin-R}
\overrightarrow{A^*}, \overrightarrow{B^*} = \operatorname*{arg\,min}_{A, B}\|\mathcal{R}(C)-\overrightarrow{A} \overrightarrow{B}\|_F^2.
\end{equation*}
$$

This directly follows from the definition of the Kronecker product, 
the definition of $$\mathcal{R}$$ and the definition of the Frobenius norm. 

This means that the closes Kronecker product to $$C$$ 
under the Frobenius norm as in Eq. \eqref{eq:argmin}, is equivalent to a rank-one matrix approximation in a permuted 
space defined by $$\mathcal{R}$$. It is straightforward to solve rank-one approximations
using e.g., a singular value decomposition of $$\mathcal{R}(C)$$. The resulting vectors 
$$\overrightarrow{A^*}$$ and $$\overrightarrow{B^*}$$ can then be re-shaped with the inverse
vectorization operation in order to obtain the closest Kronecker product 
$$A^*\otimes B^*$$ according to Eq. \eqref{eq:argmin}.

## Applications of the Kronecker product

Here are some examples of applications where the Kronecker product naturally shows up.

### Matrix normal distribution

The Kronecker product naturally occurs in the vectorized version of the 
[matrix-normal distribution](https://en.wikipedia.org/wiki/Matrix_normal_distribution).

Let $$X\in\mathbb{R}^{n_1\times n_2}$$ be a matrix-valued random variable that follows 
a matrix normal distribution with density

$$
p(X; M, U, V) = \left((2\pi)^{n_1n_2} \det(V)^{n_1}\det(U)^{n_2}\right)^{-\frac{1}{2}}
\exp{\left(-\frac{1}{2}\operatorname{tr}[V^{-1}(X-M)^{\intercal}U^{-1}(X-M)]\right)},
$$

parametrized by a mean matrix $$M\in\mathbb{R}^{n_1\times n_2}$$ 
and two positive definite matrices $$U\in\mathbb{R}^{n_1\times n_1}$$ and 
$$V\in\mathbb{R}^{n_2\times n_2}$$.
Then, its vectorized version follows a 
[multi-variate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution)
with vectorized mean matrix $$\overrightarrow{M}$$ and 
Kronecker covariance $$U\otimes V$$

$$
p(\overrightarrow{X};\overrightarrow{M}, U\otimes V).
$$

We can see this equivalence by observing that $$\operatorname{tr}[V^{-1}(X-M)^{\intercal}U^{-1}(X-M)]$$
is linked to the square of the weighted Frobenius norm $$\|X-M\|_{F, U^{-1}, V^{-1}}^2$$ and using the factorized 
version of the determinant of $$U\otimes V$$.

In this vectorized form, it is also straightforward to analyze the covariance of the
matrix-normal distribution. Since

$$
\operatorname{Cov}{[X_{ij}, X_{kl}]} = (U\otimes V)_{(ij), (kl)} = U_{ik}V_{jl}
$$

we see that the elements of $$U$$ encode how the *rows* of $$X$$ covary, while the elements of 
$$V$$ encode how the *columns* of $$X$$ covary. Hence, if e.g., we want to encode independent rows of $$X$$,
we may choose $$U = \operatorname{diag}{(u)}$$ for some vector $$u$$ or even $$U=I$$ which yields

$$
\operatorname{Cov}{[X, X']} = I\otimes V,
$$

a block-diagonal covariance matrix, with $$V$$ as blocks.

The [*symmetric* Kronecker product]({% link _posts/2022-08-06-kronecker-sym.md %}) 
is further linked to the second moment of the 
[Wishart distribution](https://en.wikipedia.org/wiki/Wishart_distribution)
which is a distribution over symmetric positive-definite matrices.


### Quasi-newton methods

Quasi-Newton methods (I may make another blogpost on those some day) 
are optimizers for deterministic objective functions
that use a local approximation to the Hessian matrix in order to obtain an approximation
to the Newton direction (this is somewhat a simplification, but the details do not matter here).
Let $$f(w)$$ be the objective function to be minimized w.r.t. $$w$$ and let $$B_t$$
be the approximation of the Hessian of $$f(w)$$ at $$w_t$$ and $$B_t^{-1}$$ its known inverse.
Then, For some initial point $$w_0$$, the update rule of a quasi-Newton method is

$$
w_{t+1} = w_t - \alpha_t B_t^{-1} \nabla f(w_t),
$$

where $$\alpha_t$$ is the step size found by a line search and 
$$B_t$$ is the analytic solution to the minimization problem

$$
B_t = \operatorname*{arg\,min}_B \|B_{t-1} - B\|_{F, W}^2
\quad\text{ s.t. the secant equation}\quad Bs_t = \Delta y_t
$$

with $$s_t = w_t - w_{t-1}$$ and $$\Delta y_t = \nabla f(w_t) - \nabla f(w_{t-1})$$.
The choice of the symmetric positive definite weight matrix $$W$$ leads to the 
different instances of quasi-Newton methods; its precise look is not important 
for the argument here.
We can already see that the vectorized version of the above equation involves 
minimization of a square form, weighted with the Kronecker product $$W \otimes W$$

$$
\overrightarrow{B}_t = 
\operatorname*{arg\,min}_{\overrightarrow{B}}
(\overrightarrow{B} - \overrightarrow{B}_{t-1})^{\intercal}
(W\otimes W)
(\overrightarrow{B} - \overrightarrow{B}_{t-1})
$$

s.t. the vectorized secant equation that also involves a Kronecker product

$$
(I\otimes s_t^{\intercal})\overrightarrow{B} = \Delta y_t.
$$

In quasi-Newton methods that enfornce symmetry of $$B_t$$, we will encounter the 
[*symmetric* Kronecker product]({% link _posts/2022-08-06-kronecker-sym.md %}) 
product in a similar way. 


### Linear algebra

Kronecker products also show up in the formulation of solvers for linear systems.
Let $$A\in\mathbb{R}^{n\times n}$$ be a matrix and $$b\in\mathbb{R}^{n}$$ a vector.
Linear  solvers solve the linear system $$Ax = b$$ for the vector $$x$$ given the solution 
to matrix-vector multiplications of the form $$A\tilde{s}_t = \Delta \tilde{y}_t$$, $$t=0, \dots$$. 
Kronecker products show up there naturally, too when vectorizing the equations. 
In fact, there are connections to quasi-Newton methods for certain types of linear systems,
hence the similar algebra.
But this, too, is a topic for another blog post.


### Deep learning 

Kronecker products show up in some stochastic optimizers that, similar to quasi-Newton methods
aim to approximate some kind of desired matrix that defines the metric of the space in which
the steepest descent is measured. 
Applying Kronecker products in that context is interesting since i) several quantities 
lend themselves to block structures or block-wise independence assumption due to the
neural network architecture, and
ii) since tensors are a natural representation of weights and gradients in most 
deep learning code bases. Hence, e.g., left an right matrix-multiplication can be thought of as
a Kronecker multiplication as in Eq. \eqref{eq:vec}.
Two recent, but not the first, examples are KFAC [[4]](#references) and Shampoo [[5]](#references), but there are
many more, also older ones that explore Kronecker structure in the context of stochastic optimization.


## References

[1] C.F. Van Loan 2000 *The ubiquitous Kronecker product*, 
    Journal of Computational and Applied Mathematics 123, pp. 85–100.

[2] C.F. Van Loan and N. Pitsianis 1993 *Approximation with Kronecker Products*,
    Linear Algebra for Large Scale and Real Time Applications. Kluwer Publications, pp. 293–314.

[3] M. Mahsereci 2018 *Probabilistic Approaches to Stochastic Optimization*, PhD thesis, Appendix A.

[4] J. Martens and r. Grosse 2015 *Optimizing Neural Networks with Kronecker-factored Approximate Curvature*, ArXiv.

[5] V. Gupta et al. 2018 *Shampoo: Preconditioned Stochastic Tensor Optimization*, ICML.

[6] P. Hennig 2015 *Probabilistic Interpretation of Linear Solvers*, SIAM.
    

