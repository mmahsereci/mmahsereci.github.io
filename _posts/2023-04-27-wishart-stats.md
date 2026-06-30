---
layout:     post
title:      "Covariance of the Wishart distribution"
author:     mmahsereci
description:    ""
date:       2023-04-27
category:   techblog
tags:       [machinelearning]
description: >
    This post contains a derivation of the covariance of the elements of a 
    Wishart distributed random matrix which can be expressed as a symmetric Kronecker product.

authors:
  - name: mmahsereci
    url: "https://github.com/mmahsereci"
    affiliations:
      name: University of T&uuml;bingen
---

The covariance of the elements of a Wishart distributed $$n\times n$$ random matrix $$A\sim\mathcal{W}_n(V, \nu)$$ 
is related to a [symmetric Kronecker product]({% link _posts/2022-08-06-kronecker-sym.md %})

$$
\mathrm{Cov}[A_{ij}, A_{kl}] = 2\nu (V\circledast V)_{(ij), (kl)},
$$

where '$$\circledast$$' denotes the symmetric Kronecker product. 
I have tried to locate derivations of the above relation, but they seem quite hard to find.
As the relation seems to pop up here and there and seems to be quite useful I did my own derivation
which this post is about.

> *Disclaimer:* No one checked my the derivation below for errors, and I was tired when writing this up, so beware of possible mistakes (though I hope it's correct).

## Preliminaries: The Wishart distribution


The support of the Wishart distribution $$\mathcal{W}_n(V, \nu)$$ are symmetric positive definite $$n\times n$$ matrices $$A$$. 
The distribution is parametrized by a positive definite $$n\times n$$ matrix $$V$$, and the degrees of freedom $$\nu>n-1.$$
Its density is

$$
  p(A) = \frac{\det(A)^{\frac{\nu-n - 1}{2}} e^{-\frac{1}{2}\mathrm{tr}(V^{-1}A)}}{ 2^{\frac{\nu n}{2}}\det(V)^{\frac{\nu}{2}} \Gamma_n\left(\frac{\nu}{2}\right)}
  \quad\text{with}\quad
  \Gamma_n\left(\frac{\nu}{2}\right) = \pi^{\frac{n(n-1)}{4}} \prod_{i=1}^n \Gamma\left(\frac{\nu}{2} - \frac{i-1}{2}\right).
$$

The random matrices $$A$$ can equally be constructed as sums of outer products of normally distributed 
$$n$$-dimensional vectors $$u\sim \mathcal{N}(0, V)$$ with covariance matrix $$V$$ according to 

$$
A = \sum_{\alpha=1}^\nu u^{\alpha} u^{\alpha\intercal}.
$$

Simply for notational convenience, I will also use the vectorized notation $$\overrightarrow{A}$$ which are the rows of $$A$$ stacked into a vector. 

## Derivation of the mean of a Wishart

To derive the mean $$ \mathbb{E}[\overrightarrow{A}]$$ we use the construction $$A = \sum_{\alpha=1}^\nu u^{\alpha} u^{\alpha\intercal}$$ with $$ u\sim \mathcal{N}(0, V)$$
as mentioned.

$$
\begin{equation*}
  \begin{split}
     \mathbb{E}[A_{ij}]
    &=  \mathbb{E}\left[\left[\sum_{\alpha=1}^\nu  u^{\alpha} u^{\alpha\intercal}\right]_{ij}\right]\\
    &=  \mathbb{E}\left[\sum_{\alpha=1}^\nu u^{\alpha}_i u^{\alpha}_j\right]\\
    &= \sum_{\alpha=1}^\nu  \mathbb{E}\left[ u^{\alpha}_i u^{\alpha}_j\right]\\
    &= \sum_{\alpha=1}^\nu  V_{ij}\\ 
    &= \nu  V_{ij}
  \end{split}
\end{equation*}
$$

The second to last line uses $$ V_{ij} = \mathrm{Cov}[ u_i,  u_j] = \mathbb{E}[ u_i u_j]  -  \mathbb{E}[u_i] \mathbb{E}[u_j] = \mathbb{E}[ u_i u_j] - 0\cdot 0 = \mathbb{E}[ u_i u_j] $$.

## Derivation of the covariance of a Wishart

Likewise for the covariance $$\mathrm{Cov}[\overrightarrow{A}]$$ we get

$$
\begin{equation*}
  \begin{split}
    \mathrm{Cov}[A_{ij}, A_{kl}]
    &=  \mathbb{E}[A_{ij}A_{kl}] -    \mathbb{E}[A_{ij}]   \mathbb{E}[A_{kl}]\\ 
    &=  \mathbb{E}[A_{ij}A_{kl}] -   \nu^2  V_{ij}  V_{kl}\\ 
    &=  \mathbb{E}\left[\left[\sum_{\beta=1}^\nu  u^{\beta} u^{\beta\intercal}\right]_{ij} \left[\sum_{\alpha=1}^\nu  u^{\alpha} u^{\alpha\intercal}\right]_{kl}\right]
    - \nu^2  V_{ij}  V_{kl}\\ 
    &=  \mathbb{E}\left[\sum_{\alpha,\beta=1}^\nu u^{\beta}_{i} u^{\beta}_{j} u^{\alpha}_{k} u^{\alpha}_{l} \right]
    -   \nu^2  V_{ij}  V_{kl}\\ 
    & = \sum_{\alpha,\beta=1}^\nu  \mathbb{E}\left[ u^{\beta}_{i} u^{\beta}_{j} u^{\alpha}_{k} u^{\alpha}_{l} \right]
    -   \nu^2  V_{ij}  V_{kl}\\ 
    & = \nu [ V_{ik}  V_{jl} +  V_{il}  V_{jk}]
     + \nu^2  V_{ij}  V_{kl}     
     - \nu^2  V_{ij}  V_{kl}\\ 
    & = \nu [ V_{ik}  V_{jl} +  V_{il}  V_{jk}] \\
    & = 2\nu (V \circledast V)_{(ij), (kl)}
   \end{split}
\end{equation*}
$$

In the second line we used the result of the mean $$\mathbb{E}[A_{il}] = \nu  V_{ij}$$.
In the third line we use the construction $$A = \sum_{\alpha=1}^\nu u^{\alpha} u^{\alpha\intercal}$$ with $$ u\sim \mathcal{N}(0, V)$$.
In the last line we used the definition of the symmetric Kronecker product 
$$(\Xi\circledast \Xi)_{(ij), (kl)} = \frac{1}{2}(\Xi_{ik}\Xi_{jl} + \Xi_{jk}\Xi_{il})$$
(see also [this]({% link _posts/2022-08-06-kronecker-sym.md %}) post). 
From fifth to sixth line we used the following derivation:

$$
\begin{equation*}
  \begin{split}
    \sum_{\alpha,\beta=1}^\nu
     \mathbb{E}[ u^{\beta}_{i} u^{\beta}_{j}  u^{\alpha}_{k} u^{\alpha}_{l}]
     & = \sum_{\alpha,\beta=1}^\nu  \mathbb{E}[ u^{\beta}_{i} u^{\beta}_{j}  u^{\alpha}_{k} u^{\alpha}_{l}] \delta_{\alpha\beta}
     + \sum_{\alpha,\beta=1, \alpha\neq\beta}^\nu  \mathbb{E}[ u^{\beta}_{i} u^{\beta}_{j}u^{\alpha}_{k} u^{\alpha}_{l}]\\
     & = \sum_{\alpha=1}^\nu  \mathbb{E}[ u^{\alpha}_{i} u^{\alpha}_{j}  u^{\alpha}_{k} u^{\alpha}_{l}]
     + \sum_{\alpha,\beta=1, \alpha\neq\beta}^\nu  \mathbb{E}[ u^{\beta}_{i} u^{\beta}_{j}]  \mathbb{E}[ u^{\alpha}_{k} u^{\alpha}_{l}]\\
     & = \sum_{\alpha=1}^\nu [ V_{ij}  V_{kl} +  V_{ik}  V_{jl} +  V_{il}  V_{jk}]
     + \sum_{\alpha,\beta=1, \alpha\neq\beta}^\nu V_{ij}  V_{kl}\\
     & = \sum_{\alpha=1}^\nu [ V_{ij}  V_{kl} +  V_{ik}  V_{jl} +  V_{il}  V_{jk}]
     + \sum_{\alpha,\beta=1}^\nu V_{ij}  V_{kl}
     - \sum_{\alpha=1}^\nu V_{ij}  V_{kl}\\
     & = \nu [ V_{ij}  V_{kl} +  V_{ik}  V_{jl} +  V_{il}  V_{jk}]
     + \nu^2  V_{ij}  V_{kl}
     - \nu   V_{ij}  V_{kl}\\
     & = \nu [ V_{ik}  V_{jl} +  V_{il}  V_{jk}]
     + \nu^2  V_{ij}  V_{kl}.
  \end{split}
\end{equation*}
$$

In the first line we split the sum into terms where $$\alpha=\beta$$ and $$\alpha\neq\beta$$.
In the second line and second term we used that $$ u^{\alpha}$$ and $$ u^{\beta}$$ were drawn independently if $$\alpha\neq \beta$$;
the first term simplifies due to the Kronecker delta $$\delta_{\alpha\beta}$$.
In the third line we use two simplifications: The first one ist the fourth moment of a multivariate Gaussian; that is if 
$$x\sim\mathcal{N}(0, \Sigma)$$ with $$\sigma_{ij}$$ the $$ij$$-th element
of the covariance matrix $$\Sigma$$ then $$\mathbb{E}[x_ix_j x_k x_l] = \sigma_{ij} \sigma_{kl} + \sigma_{ik} \sigma_{jl} + \sigma_{il} \sigma_{jk}$$. 
The second one is that $$V_{ij} = \mathrm{Cov}[ u_i,  u_j] = \mathbb{E}[ u_i u_j] $$ as already used above. Now the terms in the sums
do not depend on $$\alpha$$ and $$\beta$$ anymore, thus we only collect terms in the last three lines.

## Summary

The results for the mean, variance and covariance are hence

$$
\begin{equation*}
\begin{split}
  \mathbb{E}[A_{ij}] 
  &= \nu V_{ij} \\
  \mathrm{Var}[A_{ij}]
  &= \nu [ V_{ii}  V_{jj} +  V_{ij}^2]\\
  \mathrm{Cov}[A_{ij}, A_{kl}] 
  &= \nu [ V_{ik}  V_{jl} +  V_{il}  V_{jk}] \\
\end{split}
\end{equation*}
$$

where we also used that $$V_{ij}=V_{ji}$$.

With the definition of the symmetric Kronecker product we obtain the vectorized versions

$$
\begin{equation*}
\begin{split}
   \mathbb{E}[\overrightarrow{A}] & = \nu \overrightarrow{V}\\
  \mathrm{Cov}[\overrightarrow{A}] & = 2\nu (V\circledast V),
\end{split}
\end{equation*}
$$

where $$\overrightarrow{A}$$ is an $$n^2$$-dimensional vector as mentioned (rows of $$A$$ are stacked),
and the associated covariance matrix is a symmetric Kronecker product of shape $$n^2\times n^2$$.

