---
layout:     post
title:      "Dataset Shifts"
author:     mmahsereci
description:    ""
date:       2021-10-31
category:   techblog
tags:       [machinelearning, statistics]
description: >
  Distinguishing data-distributions is a central topic of statistics. There were several attempts to categorize the 
  different ways and causes to why two data-distributions, in particular the training and test distribution, may differ. 
  Here, I collect some thoughts on the issue.

authors:
  - name: mmahsereci
    url: "https://github.com/mmahsereci"
    affiliations:
      name: University of T&uuml;bingen
---

Distinguishing data-distributions is a central topic of statistics. There were several attempts to categorize the 
different ways and causes to why two data-distributions, in particular the training and test distribution, may differ. 
Here, I collect some thoughts on the issue.

The term *dataset shift* was probably coined by [Storkey 2009](#references) and means that the training and 
test datasets in a supervised learning problem differ not solely due to the randomness of the data sample, 
but also due to the test and 
training data-distribution having different sample space and/or probability of events. 
Thus, dataset shifts occur when models are fitted to training data that does not reflect the data-distribution they 
encounter once deployed.
Two of the most frequent reasons for dataset shifts are:

- *Sample selection bias*: Samples may be discarded from the training with a certain, possibly unknown probability, 
- *Changing environments*: The mechanism that produces the data changes between collecting 
  the training data and deployment.

Specific instances of dataset shift where categorised by [Moreno-Torres et al. 2012](#references) in a thorough review.
Probably the most well-known shift is the *covariate shift* where the marginal distribution of the inputs changes, but the conditional 
distribution of the targets given the inputs stays unchanged. 
We'll now explore the shift types based on which marginal or conditional distribution is acted upon.


## The 4 types of dataset shift

Consider a supervised prediction problem (classification or regression) where a relation between inputs 
$$x$$ and targets $$y$$ needs to be learned. Denote the available dataset at training time as 
$$\mathcal{D}_{tr} :=\{(x_n, y_n)\}_{n=1}^N$$ with elements assumed to be iid draws from a data-distribution 
$$(x_n, y_n)\sim P_{tr}(x, y)$$. 
Denote the dataset which will be accesible at deployment time as $$\mathcal{D}_{ts}=\{(x_m, y_m)\}_{m=1}^{M}$$ with 
$$(x_m, y_m)\sim P_{ts}(x, y)$$.

In machine learning, the distributions at training and deployment time $$P_{tr}$$ and $$P_{ts}$$ respectively are 
often implicitly assumed to be 
identical, in which case we can assess generalisation performance at training time already, e.g., by splitting off 
a test set from $$\mathcal{D}_{tr}$$ and witholding it during training.
In practical applications, the implicit assumption of $$P_{tr}\equiv P_{ts}$$ may be violated, and the distribution 
$$P_{ts}$$ at deployment time may be arbitrarily different to the distribution $$P_{tr}$$ at fitting time. 
This 'difference in data-distribution' is often referred to as *dataset shift*. There are 
*4 types of shifts* which each describe a special case where one distribution of the data-generating process 
stays fixed, and the other one changes. The following table summarises those shifts. 

| name                      |   causal direction   |   &nbsp; &nbsp; marginal probability   &nbsp; &nbsp;  | &nbsp; &nbsp;  conditional probability  |
|---------------------------|:--------------------:|:-----------------------------------:| --------------------------------------------:|
|  covariate shift          | $$X\rightarrow Y$$   |  $$P_{tr}(x)\not\equiv P_{ts}(x)$$  | $$P_{tr}(y\vert x)\equiv P_{ts}(y\vert x)$$ |
|  concept shift i          | $$X\rightarrow Y$$   |  $$P_{tr}(x)\equiv P_{ts}(x)$$      | $$P_{tr}(y\vert x)\not\equiv P_{ts}(y\vert x)$$ |
|  prior probability shift  |  $$Y\rightarrow X$$  |  $$P_{tr}(y)\not\equiv P_{ts}(y)$$  | $$P_{tr}(x\vert y)\equiv P_{ts}(x\vert y)$$ |
|  concept shift ii         |  $$Y\rightarrow X$$  |  $$P_{tr}(y)\equiv P_{ts}(y)$$      | $$P_{tr}(x\vert y)\not\equiv P_{ts}(x\vert y)$$ |

<br>
Two of the shifts (covariate and prior probability shift) act on one of the marginal distributions each, 
while the other two (concept shift i and ii) act on one of the conditional distributions while the 'other' 
distribution completing the joint stays the same for all types.

The shifts include the causal direction of the data generation in their definitions. 
Dataset shift is thus an intervention into the system, where either the generation of inputs $$x$$ via $$P(x)$$ 
(forward direction $$X\rightarrow Y$$), the generation of the targets $$y$$ via $$P(y)$$ 
(inverse direction $$Y\rightarrow X$$), or the mechanisms $$P(y|x)$$ and $$P(x|y)$$ that create the targets and inputs 
respectively, are being altered.
The causal relation ensures that if one of the 4 distributions is intervened on---one of the 2 marginals 
$$P(x)$$ and $$P(y)$$, or one of the 2 conditionals $$P(x|y)$$ and $$P(y|x)$$---the other relevant distribution 
completing the joint does not change.

This also means that the 4 shift types do not cover the entirety of possible dataset shifts as there are more causal 
structures in practice than the ones considered above.
The dataset shifts that fall under the 4 types in practice are probably just a small sub-set of 
possible dataset shifts one might encounter as practitioner; and, unless we know the ground truth, it's probably really 
hard to find out what the causal relation between $$x$$ and $$y$$ is.

Granted there is quite a bit of confusion about naming in the dataset shift literature, 
even authors assigning the same name to opposing 
concepts, but nevertheless (or rather therefore) it is beneficial to be clear on definitions.
The ones above are taken from [Moreno-Torres et al. 2012](#references).

#### The colloquial use of the term 'covariate shift'

This is a bit of a tangent (skip to the next section if you like), 
but did you ever read in a machine learning paper that it addresses covariate shift? 
Probably that's not what they are doing.
As we've just seen, covariate shift is the change of the marginal distribution $$P(x)$$ of features $$x$$ assuming the causal 
relation $$X\rightarrow Y$$, and $$P(y|x)$$ stays unchanged when the intervention (shift) occurs.
The covariate shifted dataset follows the generative process $$x\sim P_{ts}(x)$$, $$y\sim P(y|x)$$ while the 
original dataset follows $$x\sim P_{tr}(x)$$, $$y\sim P(y|x)$$.
Quite frequently, any  statistical change in the marginal distribution $$P(x)$$ of features is coined as 
covarite shift, disregarding the distribution $$P(y|x)$$. 

Hence, the term 'covariate shift' is often used colloquially in the machine learning literature when in fact, 'any'
dataset shift is meant, and the actual shift is not specified, and sometimes not well understood. 

*Example MNIST:*
The causal direction of data-generation for MNIST is inverse ($$Y$$ causes $$X$$). Printed forms were handed to 
volunteers ([example](https://www.nist.gov/srd/nist-special-database-19) of a filled form). The forms contained boxes 
in which the volunteers were supposed to draw pre-defined number-characters. 
Thus, the labels $$y\sim P(y)$$ were generated first by printing and distributing the forms. Then, the features 
$$x\sim P(x|y)$$ were produces by the volunteers, digitalisation and post-processing.<br>
It is common practice in deep learning to "shift" datasets by transforming the features $$x$$.
But, for this shifted/transformed image $$\tilde{x}$$ derived from $$x$$, it is not possible to attach a label that 
follows the original data generating process because there is no such process.
Strictly speaking, transforms on the features $$x$$ for MNIST fall under concept shift of type ii as the 
transformation changes the process underlying 
$$P(x|y)$$ given some $$y$$, while the process underlying $$P(y)$$ stays intact.



## How does dataset shift look like?

#### Each type of dataset shift can have different appearances
Let's assume the dataset shift we are considering falls into one of the 4 shift type categories as defined above.
The definitions merely state which of the 4 
distributions $$P(x)$$, $$P(y|x)$$, $$P(y)$$ and $$P(x|y)$$ changes based on the intervention on the underlying 
data generating process. The definitions do not state how this change will look like.

Thus, dataset shift, even of the same type, can have wildly different appearance.
Common realizations are drifting distributions where the dataset shift happens gradually over time, abrupt changes 
in the distribution where the data generating process is shifted at discrete points in time, 
or periodic shifts where the data generating process returns to one of it's states cyclically.
Thus, simply by looking at the data, we may not know the dataset shift; but also, and perhaps more importantly,
how a practitioner handles dataset shift may depend on the appearance rather than the type, or possibly on
their combination.


#### Each type of dataset shift can have different root causes 

The different appearances can be explained by the root causes of the dataset 
shift, i.e., the reason why the dataset shift happens. There are two common root causes. The first one is:

- *Changes in the environment*.

Changes in the environment can lead to gradual, cyclic, or abrupt changes and can occur for any dataset shift type. 
The root cause ist not equivalent to the shift types. 
The types explain which distribution is affected but not why it is affected.
Thus, the root cause tells us why the shift happens, the type tells us which distribution is intervened on, and the 
appearance tells use what the shift looks like. 

In addition to changing environments, a second common root cause is:

- *Sample selection bias*.
 
Roughly, sample selection bias means that there is a mechanism that reduces the probability of a sample being included 
in the dataset.
The training distribution may not be represented well if datapoints are selected not according to $$(x,y) \sim P_{tr}(x,y)$$ but 
according to $$(x,y) \sim P_{tr}(x,y|s=1)$$ where $$s$$ is a binary selection variable and $$Q(s = 1|x, y)$$ is the 
probability of accepting $$(x, y)$$ into the training dataset. Sample selection bias is zero if $$Q(s = 1|x, y) = 1$$ 
(see [Quionero-Candela et al. 2009](#references), Section 3.2 for formulas and conditional distributions). 

Unwanted sample selection bias can occur for example when the environment where the training was 
collected does not capture all aspects of the test environment. An example is training data that consists of 
measurements of a vehicle in a wind-tunnel. The wind-tunnel cannot produce certain wind configurations $$x$$ that 
appear in the real world where it is tested. For those configurations $$Q(s = 1|x, y) = 0$$.

There is a lot more to say about sample selection bias, and this may be a topic for another blog post, but for now, 
we only need to remember that it may induce dataset shift.

## What's the conclusion?

Does dataset shift matter? In real world applications, it's probably hard to know for sure what dataset shift type occurs, but even if we did 
know does that information provide any benefit? I guess it's not entirely clear, but here are some thoughts 
(which should be taken with a grain of salt, as this is my current take only):

- *The type of shift often shouldn't matter:* A predictive model probably does not care
  a lot if the statistical change in $$P(x)$$ was causes by covariate shift, inverse concept shift, prior probability 
  shift, or some undefined shift. All that matters is that the input of the learner changes in 
  some way such that the pattern in $$P(y|x)$$, and hence the learned representation, is different now. This could be due to exploring a different domain 
  of $$x$$, which can be caused by several shift types, or because the mechanism connecting $$x$$ with $$y$$ 
  changed. So what really matters is that the learner either i) sees a lot of data from all possible scenarios, 
  or ii) somehow understands that it deals with a new scenario. The former is hard to do by default. 
  How the latter can be achieved is an open question
  as it is not straightforward to estimate if and in what way a shift implies that the learned patterns do not apply anymore. 
  Hence, the machine learning field has worked on data-driven approaches such as meta-learning or continual learning, 
  where the jury is still out if they are applicable and robust in practice. Bayesian models are promising as well.
  
- *The type of shift should matter:* It is often not clear
  or easily interpretable how a trained machine learning model reacts to dataset shift. Hence, if we want to 
  use models in the real world, we should care. Possibly increasingly in benchmarks, where shifts are 
  often ad-hoc and definitions are not provided or incorporated in the analysis. Sometimes there are even discrepancies 
  between shifts used in toy-examples that are meant to build intuition and so-called real world applications 
  which makes interpretation of experimental results even harder.

One approach supposed to combat dataset shift is meta-learning I may do a blog post on it at some point :)

## References

[1] Storkey A.J. 2009 *When Training and Test Sets are Different: Characterising Learning Transfer*,
    in *Dataset Shift in Machine Learning* MIT Press, pages 3-28.

[2] Moreno-Torres et al. 2012 *A unifying view on dataset shift in classification*, 
    Pattern Recognition 45, pages 521–530.

[3] Quionero-Candela et al. 2009 *Dataset Shift in Machine Learning*,
    The MIT Press.


<br>

---

## Appendix: Some further observations

You scrolled too far! But since you're here, this is a somewhat random collection of thoughts
that did not quite make it into the blog post above, so bear with me.

#### Is dataset shift a reliable indicator for change in performance?

We are getting into the wild territory of applied research here. But let's just roll with it for now.
What if I deployed a performant model, do I need to worry about dataset shift? The answer is probably yes, but it is not so simple.
Consider for instance a supervised prediction problem and  a machine learning model trained on a dataset 
$$\mathcal{D}_{tr}=\{(x, y)_n\}_{n=1}^N$$ with elements i.i.d. draws from some data distribution 
$$(x, y)\sim P_{tr}(x, y)$$ 
($$\mathcal{D}_{tr}$$ might be split up in a train/validation/test split during training). 
Consider a dataset shift at deployment time such that $$\mathcal{D}_{ts} = \{(x, y)_m\}_{m=1}^M$$ 
with $$(x, y)_m\sim P_{ts}(x, y)$$ and $$P_{ts}(x, y)\neq P_{tr}(x, y)$$.
Also, suppose we have access to a metric or score $$S(\mathcal{D}_{tr}, \mathcal{D}_{ts})$$ that quantifies 
reliably the dataset shift between training 
and deployment time (we leave aside the issue for now that it is unclear how to define this score).
Is this score a good indicator for predictive model performance on the shifted dataset? 
This question is important as score metrics could be used to inform re-training decisions, coresets, or other 
learning parameters of an already trained model.
The fast answer is that it's not so easy to say. We won't go into too much detail in this blog post appendix, 
but here are some reasons:

*Feature importance:* Consider e.g., covariate shift. If an unimportant feature shifts, the generalisation 
will not be affected much. If an important features shifts, the generalisation performance will drop. A shift score 
would give the same signal. Likewise, if an important features shifts only a litte this may impact performance 
disproportionately high, which is not reflected in the value of the score. 

*Shift type:* During deployment, often only $$P(x)$$ is available for a new dataset, and the targets $$y$$ are missing.
Hence, the score $$S$$ needs to be computed on $$P_{ts}(x)$$ only.
Covariate shift, inverse concept shift and to some degree also prior probability shift all 
will have a detectable effect on the statistics of $$P_{ts}(x)$$. If targets $$y$$ are 
not available in a new dataset however, and forward concept shift occurs ($$P(y|x)$$ is intervened on), then no 
shift will be detected, but generalisation performance will drop. This scenario is not too unrealistic: A root cause
example are changes in the environment where users start to like or not like a feature anymore due to some outer 
influence.

*Model assumptions:* There is an interplay between dataset, shift appearance, and machine learning model. 
A relatively small shift may have a larger effect on generalisation performance, e.g., swapping pixels in an image, than 
a larger shift that keeps the smooth structure of an image intact. 
An example are neural networks that encode prior assumptions 
about what patters they expect to see such as the design of filters in convolutional neural networks. 
If this assumption is broken by the shift, even small 
shifts can have a large effect on performance, and likewise large shifts can have less effect if those assumptions are not violated.

The above list is by no means complete.

