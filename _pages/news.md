---
layout: page
permalink: /news/
title: radio
description: 
nav: false
---


<table class="table table-sm table-borderless">
{% assign news = site.news | reverse %}
{% for item in news limit: 1000 %}
<tr>
  <th scope="row">{{ item.date | date: "%b %-d, %Y" }}</th>
  <td>
    {% if item.inline %}
      {{ item.content | remove: '<p>' | remove: '</p>' | emojify }}
    {% else %}
      <a class="news-title" href="{{ item.url | relative_url }}">{{ item.title }}</a>
    {% endif %}
  </td>
</tr>
{% endfor %}
</table>


### 2016

- 12/2016: I **co-organized** the workshop [*Optimizing the Optimizers*](http://www.probabilistic-numerics.org/en/latest/research/meetings/NIPS2016.html) @ NIPS 2016, Barcelona, Spain.
- 09/2016: I gave a **talk** on *Stochastic Optimization* at the Workshop on Uncertainty Quantification @ [*Gaussian Process Summer School*](http://gpss.cc/gpuqss16/)
  (GPSS) Sheffield, UK.
- 07/2016: **I joint the Amazon.com** as Applied Science Intern for three month for a **summer internship** @ *Amazon Development Center* in Berlin, Germany.
 

### 2015 

- 12/2015: Our **paper** [*Probabilistic Line Searches for Stochastic Optimization*](https://proceedings.neurips.cc/paper/2015/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf) 
  has been accepted @ NIPS 2015 and selected for a **full oral presentation** (acceptance <1%).
- 10/2015: Looking forward to work with our lab rotation student *Jonas Rauber*.
- 09/2015: I gave a **talk** on *Line Searches for Stochastic Optimization* at Microsoft Research - Cambridge, UK.
- 05/2015: Looking forward to working with our student assistant *Jonas Jaszkowic* on some code for probabilistic numerics.
- 04/2015: I gave a **talk** on *Line Searches and Stochastic Optimization* at the Workshop on Probabilistic Numerics 
  @ [*Data, Learning and Inference*](http://dalimeeting.org/dali2015/) (DALI) La Palma, Canary Islands. 
- 07/2015:I attended the [*Machine Learning Summer School*](http://mlss.tuebingen.mpg.de/2015/) (MLSS) in T&uuml;bingen, Germany.

### before

- 08/2014: I attended the [*Roundtable on Probabilistic Numerics*](http://www.probabilistic-numerics.org/en/latest/research/meetings/RoundtablePN2014.html) @ 
  Max Planck Institute (MPI) for Intelligent Systems, T&uuml;bingen, Germany.
- 08/2013:I attended the [*Machine Learning Summer School*](http://mlss.tuebingen.mpg.de/2013/2013/index.html) (MLSS) in T&uuml;bingen, Germany.
- 07/2013: **I joint the Max Planck Institute (MPI) for Intelligent Systems** in T&uuml;bingen, Germany, as a PhD candidate in machine learning. 
- 06/2013: I attended the [*Gaussian Process Summer School*](http://gpss.cc/gpss13/) (GPSS) as well as the 
  [*Latent Force Model Workshop*](http://gpss.cc/lfm13/) Sheffield, UK.
