---
layout: page
permalink: /bq-book/exercises/
title: Exercises for BQ Book
nav: false
sitemap: false
---

{% comment %}
HOW TO ADD CONTENT (this block is not rendered)

Exercise numbering: N.M(x) = chapter N, exercise M, part x. Headings get an
explicit id {#ex-N-M}, which solution pages use for their back-link.

Exercise template (parts, hints and solution link are all optional):

### Exercise N.M {#ex-N-M}

Shared setup text.

**(a)** Question for part a.

<details class="hint" markdown="1">
<summary>Hint</summary>
Hint text, may contain math $$x^2$$.
</details>

**(b)** Question for part b.

[Solution →](/bq-book/exercises/solutions/N-M/)

Solutions: copy an existing file in the solutions/ folder next to this file, rename it to
N-M.md, and update title, permalink and back-link. Then add the
"Solution →" link to the exercise above.
{% endcomment %}

**The book:** [Book (link 1)](#) · [Book (link 2)](#)

**How to use this page.** <span class="text-danger"><strong>Disclaimer:</strong> this page is under development and so far only contains dummy exercises.</span> Placeholder text: all exercises are listed below, ordered by chapter. Some parts come with a hint, which you can expand by clicking on it. Where a solution is available, a link at the end of the exercise leads to it.

## Chapter 1

### Exercise 1.1 {#ex-1-1}

Let $$f(x) = x^2 - 2x + 1$$.

**(a)** Show that $$f$$ is convex.

<details class="hint" markdown="1">
<summary>Hint</summary>
Compute the second derivative $$f''(x)$$.
</details>

**(b)** Find the minimizer $$x^\ast$$ of $$f$$.

[Solution →](/bq-book/exercises/solutions/1-1/)

### Exercise 1.2 {#ex-1-2}

Show that the sum of two convex functions is convex.

<details class="hint" markdown="1">
<summary>Hint</summary>
Use the definition $$f(\lambda x + (1-\lambda) y) \leq \lambda f(x) + (1-\lambda) f(y)$$ for both functions.
</details>

## Chapter 2

### Exercise 2.1 {#ex-2-1}

Draw $$n = 1000$$ samples from a standard normal distribution and estimate $$\mathbb{E}[X^2]$$ by Monte Carlo.

[Solution →](/bq-book/exercises/solutions/2-1/)

### Exercise 2.2 {#ex-2-2}

Let $$X \sim \mathcal{N}(\mu, \sigma^2)$$.

**(a)** Compute $$\mathbb{E}[X]$$.

**(b)** Compute $$\mathrm{Var}[X]$$.

<details class="hint" markdown="1">
<summary>Hint</summary>
Use $$\mathrm{Var}[X] = \mathbb{E}[X^2] - \mathbb{E}[X]^2$$.
</details>

**(c)** Compute $$\mathbb{E}[e^{X}]$$.
