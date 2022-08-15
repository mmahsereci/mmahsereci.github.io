---
layout:     post
title:      "The Symmetric Kronecker Product"
author:     mmahsereci
description:    ""
date:       2022-08-06
category:   techblog
tags:       [machinelearning]
description: >
    The symmetric Kronecker product can be derived from the Kronecker product. 
    It again appears naturally in some machine learning applications.
    This post also discusses the lesser known anti-symmetric Kronecker product
    for completeness.

authors:
  - name: mmahsereci
    url: "https://github.com/mmahsereci"
    affiliations:
      name: University of T&uuml;bingen
---
  
> This is a follow-up post to [this]({% link _posts/2022-08-05-kronecker.md %}) post about the Kronecker product.
> Notation and definitions will be used from the previous post.

The symmetric Kronecker product can be derived from the Kronecker product. 
It again appears naturally in some machine learning [applications](#applications-of-the-kronecker-product).
This post also discusses the lesser known 
[anti-symmetric Kronecker product](#the-anti-symmetric-kronecker-product)
for completeness.

Let $$\Gamma\in\mathbb{R}^{n^2\times n^2}$$ be a 
projection operator implicitly defined such that

$$
\Gamma \overrightarrow{C} = \frac{1}{2}(\overrightarrow{C + C^{\intercal}})
$$

symmetrizes the un-vectorized matrix $$C\in\mathbb{R}^{n\times n}$$. 
Let $$A\in\mathbb{R}^{n\times n}$$ and $$B\in\mathbb{R}^{n\times n}$$ be square matrices
of same size. 
Then, the symmetric Kronecker product can be defined as

$$
A \circledast B = \Gamma(A\otimes B)\Gamma^{\intercal}
$$

with elements

$$
(A\circledast B)_{(ij), (kl)} 
= \frac{1}{4}(A_{ik}B_{jl} + A_{il}B_{jk} + A_{jk}B_{il} + A_{jl}B_{ik}).
$$

## Symmetric Kronecker algebra

Like to the Kronecker product, the symmetric Kronecker product has some nice
algebraic properties which are similar but differ in key aspects to the properties of the Kronecker product.

$$
\begin{alignat}{3}
&\text{transpose}           &(A\circledast B)^{\intercal} &= (A^{\intercal}\circledast B^{\intercal}) && \\
\label{eq:3-sym}
&\text{inverse}             &(A\circledast A)^{-1} &= A^{-1}\circledast A^{-1} &&\quad\text{but}\quad (A\circledast B)^{-1} \neq A^{-1}\circledast B^{-1} \\
&\text{factorizing}\quad         &(A\circledast A)(C\circledast C)&= (AC\circledast AC) &&\quad\text{but}\quad (A\circledast B)(C\circledast D) \neq (AC\circledast BD)
\end{alignat}
$$

The symmetric Kronecker product factorizes according to 

$$
(A\circledast B)(C\circledast D)= \frac{1}{2}[AC\circledast BD + AD\circledast BC].
$$

In contrast to the Kronecker product, the symmetric Kronecker product commutes

$$
A\circledast B = B\circledast A
$$

and its trace is given by

$$
\operatorname{tr}[(A\circledast B)] 
= \frac{1}{2}(\operatorname{tr}[A]\operatorname{tr}[B]  
+ \operatorname{tr}[AB]).
$$

It is worth noting that

$$
I_n\otimes I_n = I_{n^2}\quad\text{but}\quad I_n\circledast I_n = \Gamma \neq I_{n^2}.
$$


## Vector multiplication

A similar formula to Eq. (8) of the 
[previous post]({% link _posts/2022-08-05-kronecker.md %})
holds for the symmetric Kronecker product

$$
(A\circledast B)\overrightarrow{X} = \frac{1}{4}
\overrightarrow{AXB^{\intercal} + AX^{\intercal}B^{\intercal} + BX^{\intercal}A^{\intercal} + BXA^{\intercal}}.
$$

For the special case of $$A=B$$ the equations simplify to 

$$
\begin{align*}
(A\circledast A)_{(ij), (kl)} 
&=\frac{1}{2}(A_{ik}A_{jl} + A_{jk}A_{il})\\
(A\circledast A)\overrightarrow{X} 
&=\frac{1}{2}\overrightarrow{AXA^{\intercal} + AX^{\intercal}A^{\intercal}}.
\end{align*}
$$


## Closest symmetric Kronecker product

The closest symmetric Kronecker product $$\tilde{A}^*\circledast\tilde{B}^*$$ under the Frobenius norm is defined as

$$
\begin{equation}
\label{eq:argmin-sym}
\tilde{A}^*, \tilde{B}^* = \operatorname*{arg\,min}_{A, B}\|C-A\circledast B\|_F^2.
\end{equation}
$$

Since there exists a fixed linear operator $$\mathcal{T}$$ ([[3]](#references), Eq. 199)
such that (with slight abuse of notation)

$$
\begin{equation*}
\tilde{A}^*, \tilde{B}^*  = \operatorname*{arg\,min}_{A, B}\|\mathcal{T}[C-A\otimes B]\|_F^2.
\end{equation*}
$$

Eq. \eqref{eq:argmin-sym} can be solved by solving the closest Kronecker problem (Eq. (11) of the
[previous post]({% link _posts/2022-08-05-kronecker.md %}))
instead and then the solution can be symmetrized according to 

$$
\begin{equation*}
\begin{split}
\tilde{A}^* &= A^*\quad\text{and}\quad \tilde{B}^* =B^*\\
\tilde{A}^*\circledast \tilde{B}^*
& = \mathcal{T}[A^*\otimes B^*]
=A^*\circledast B^*.
\end{split}
\end{equation*}
$$

## Applications of the symmetric Kronecker product

The [previous post]({% link _posts/2022-08-05-kronecker.md %})
already discussed some applications of the Kronecker product. Here we 
mention some applications where the use of the symmetric Kronecker product is key
or naturally shows up. Of course the list is not complete.

### Symmetric matrix normal distribution

Consider a matrix-valued random variable $$X\in\mathbb{R}^{n_1\times n_2}$$  that follows a
[matrix-normal distribution](https://en.wikipedia.org/wiki/Matrix_normal_distribution).
We already know from the [previous post]({% link _posts/2022-08-05-kronecker.md %})
that the vectorized version of $$\overrightarrow{X}$$ follows the multi-variate normal distribution

$$
p(\overrightarrow{X};\overrightarrow{M}, U\otimes V),
$$

where $$M$$ is the mean matrix, and $$V$$ and $$U$$ are symmetric positive definite matrices
that parametrize the matrix-normal distribution.


For square $$X\in\mathbb{R}^{n\times n}$$, the domain can be naturally restricted to only 
allow symmetric matrices by applying the symmetrization operator $$\Gamma$$ 
to the random variable $$\overrightarrow{X}$$

$$
\begin{equation}
\label{eq:pdf-sym}
\begin{split}
\overrightarrow{X}_{s} &= \Gamma\overrightarrow{X}
\quad\text{and hence}\quad\\
p(\overrightarrow{X}_{s}; \overrightarrow{M}_{s}, U\circledast V)
&= p(\Gamma\overrightarrow{X}; \Gamma \overrightarrow{M}, \Gamma(U\otimes V)\Gamma^{\intercal}).
\end{split}
\end{equation}
$$

Above, we used the definition of the symmetric Kronecker product and 
the closeness property of the normal distribution under linear transformations. 
Hence, a distribution over symmetric matrices $$X_s$$
can be achieved by assuming a symmetric mean matrix $$M_s$$
and a symmetric Kronecker product as covariance for its vectorized form.

However, since $$(U\circledast V)^{-1} \neq U^{-1}\circledast V^{-1}$$ in general 
and only equal if $$U=V$$ (Eq. \eqref{eq:3-sym}), it is often prudent to restrict to 
$$U=V$$ when working with Eq. \eqref{eq:pdf-sym}.


### Wishart distribution

The symmetric Kronecker product also naturally occurs in the 
[Wishart distribution](https://en.wikipedia.org/wiki/Wishart_distribution)
over symmetric positive-definite matrices. 

Let $$X\in\mathbb{R}^{n\times n}$$ be a matrix-valued random variable that follows a 
[Wishart distribution](https://en.wikipedia.org/wiki/Wishart_distribution)
with density

$$
p(X; V, \nu) = \det(X)^{\frac{\nu-n-1}{2}}
\frac{e^{-\frac{1}{2}\operatorname{tr}[V^{-1}X]}}{2^{\frac{\nu n}{2}}\det(V)^{\frac{\nu}{2}}\Gamma_n(\frac{\nu}{2})},
$$

that is parametrized by the scalar degrees of freedom $$\nu>n-1$$ and a symmetric positive definite matrix $$V$$.
Further, $$\Gamma_n$$ is the multivariate Gamma function. Then, it can be shown straightforwardly that
the covariance (centered second moment) of $$X$$ is given by the elemetns of the symmetric Kronecker product 
of $$V$$ scaled with a constant factor of $$2\nu$$ 

$$
\operatorname{Cov}{[X_{ij}, X_{kl}]} = 2\nu (V\circledast V)_{(ij), (kl)}.
$$


## References

[1] C.F. Van Loan 2000 *The ubiquitous Kronecker product*, 
    Journal of Computational and Applied Mathematics 123, pp. 85–100.

[2] C.F. Van Loan and N. Pitsianis 1993 *Approximation with Kronecker Products* 
    Linear Algebra for Large Scale and Real Time Applications. Kluwer Publications, pp. 293–314.

[3] M. Mahsereci 2018 *Probabilistic Approaches to Stochastic Optimization*, PhD thesis, Appendix A.

---

## Appendix: The anti-symmetric Kronecker product

I am not actually sure if the anti-symmetric Kronecker product is a thing, or if it has
been used anywhere, but since it is the counter-part to the well-established symmetric
Kronecker product it is here for completeness ([[3]](#references) Appendix A.3).

First, define the anti-symmetrization operator 
$$\Delta\in \mathbb{R}^{n^2\times n^2}$$
as the counter-part to the symmetrization operator $$\Gamma$$ such that

$$
\Delta = I + \Gamma\quad \text{and}
\quad
\Delta \overrightarrow{X} = \frac{1}{2}(\overrightarrow{X - X^{\intercal}})
$$

implicitly projects onto the un-vectorized anti-symmetric part of the matrix $$X$$.

Let $$A\in\mathbb{R}^{n\times n}$$ and $$B\in\mathbb{R}^{n\times n}$$ be square matrices
of same size. 
Then, the anti-symmetric Kronecker product can be defined as

$$
A \circleddash B = \Delta (A\otimes B) \Delta^{\intercal},
$$

with elements

$$
(A\circleddash B)_{(ij), (kl)} 
= \frac{1}{4}(A_{ik}B_{jl} - A_{il}B_{jk} - A_{jk}B_{il} + A_{jl}B_{ik}),
$$

### Decomposition of the Kronecker product in symmetric and anti-symmetric part

The Kronecker product decomposes as

$$
A\otimes B = A\circledast B + A\circleddash B
+ \Delta(A\otimes B)\Gamma^{\intercal}
+ \Gamma(A\otimes B)\Delta^{\intercal}
$$

For the special case of $$A=B$$, we get
$$\Delta(A\otimes A)\Gamma^{\intercal} = \Gamma(A\otimes A)\Delta^{\intercal}=0$$
and $$A\otimes B$$ fully decomposes into a symmetric and anit-symmetric part 

$$
A\otimes A = A\circledast A + A\circleddash A.
$$

If $$A\otimes A$$ has full rank of $$n^2$$, then the symmetric Kronecker
product and anti-symmetric Kronecker product span the $$\frac{1}{2}n(n+1)$$
and $$\frac{1}{2}n(n-1)$$ dimensional symmetric and anti-symmetric subspace respectively.

### Vector multiplication

The vectorization equation for the anti-symmetric Kronecker product is 

$$
(A\circleddash B)\overrightarrow{X} = \frac{1}{4}
\overrightarrow{
AXB^{\intercal} 
- AX^{\intercal}B^{\intercal} 
- BX^{\intercal}A^{\intercal} 
+ BXA^{\intercal}}.
$$

### Anti-Symmetric Kronecker algebra

The properties of the anti-symmetric Kronecker product mimic the ones of the 
symmetric Kronecker product

$$
\begin{alignat*}{3}
&\text{transpose}           &(A\circleddash B)^{\intercal} &= (A^{\intercal}\circleddash B^{\intercal}) && \\
&\text{inverse}             &(A\circleddash A)^{-1} &= A^{-1}\circleddash A^{-1} &&\quad\text{but}\quad (A\circleddash B)^{-1} \neq A^{-1}\circleddash B^{-1} \\
&\text{factorizing}\quad    &(A\circleddash A)(C\circleddash C)&= (AC\circleddash AC) &&\quad\text{but}\quad (A\circleddash B)(C\circleddash D) \neq (AC\circleddash BD)
\end{alignat*}
$$

The ant-symmetric Kronecker product factorizes according to 

$$
(A\circleddash B)(C\circleddash D)= \frac{1}{2}[AC\circleddash BD + AD\circleddash BC].
$$

Like the symmetric Kronecker product, the anti-symmetric one also commutes

$$
A\circleddash B = B\circleddash A
$$

and its trace is given by

$$
\operatorname{tr}[(A\circleddash B)] 
= \frac{1}{2}(\operatorname{tr}[A]\operatorname{tr}[B]  
- \operatorname{tr}[AB]).
$$

### Application: Anti-symmetric matrix normal distribution


Consider a matrix-valued random variable $$X\in\mathbb{R}^{n\times n}$$ that follows a 
matrix normal distribution with mean matrix $$M$$ and symmetric positive definite 
matrices $$U$$ and $$V$$ that parametrize the distribution.

Analogously to above, for anti-symmetric matrices, the anti-symmetrization operator $$\Delta$$ 
and the anti-symmetric Kronecker product, we can restrict the domain and write

$$
\begin{equation}
\label{eq:pdf-asym}
\begin{split}
\overrightarrow{X}_{a} &= \Delta\overrightarrow{X}
\quad\text{then}\quad\\
p(\overrightarrow{X}_{a}; \overrightarrow{M}_{a}, U\circleddash V) 
&= p(\Delta\overrightarrow{X}; \Delta \overrightarrow{M}, \Delta(U\otimes V)\Delta^{\intercal}).
\end{split}
\end{equation}
$$
