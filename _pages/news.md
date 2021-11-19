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
