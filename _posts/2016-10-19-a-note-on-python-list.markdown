---
layout: post
title:  "A Note on Python List"
date:   2016-10-19 13:44:00 +0800
categories: jekyll update
---

Created by Richard Jiang on 19th, Oct, 2016. 

I came across this interesting (or silly, given my own understanding and skills of Python) bug when using List in Python. Consider this code snippets:

{% highlight Python %}
### pts is a 11x3 numpy array
### findHomo() is a function doing processing
u = range(0, 4)
results = []
for point in pts:
	u[0] = findHomo(point, frame1)
	u[1] = findHomo(point, frame2)
	u[2] = findHomo(point, frame3)
	u[3] = findHomo(point, frame4)
	results.append(u)

print results
{% endhighlight %}

The idea is to process the element in pts one  by one and store the result inside the results[] List. By a glance of the code it looks that the logic is fine, but if you really run the code then you'll notice that: all 11 elements in results will be the same!

Therefore, it appears that here when appending elements to the List from another List, one thing to note is the reference-or-value problem. The correct way to write it is:

{% highlight Python %}
results = []
for point in pts:
	u = range(0, 4)
	u[0] = findHomo(point, frame1)
	u[1] = findHomo(point, frame2)
	u[2] = findHomo(point, frame3)
	u[3] = findHomo(point, frame4)
	results.append(u)
{% endhighlight %}

Or, alternatively, (I've not actually tried it out, but this should be working) use u.copy() as the element to be appended. A similar type of question can be found here: [Python reference or value][python-reference-or-value], and according to it, the built-in copy() function should be able to tackle the problem.

This is my first technical blog related to one of my assignments in computer vision, and I got a few bugs in the code which are not easy to solve; this is one of them, the other is related to numpy multi-dimensional array, referring to this: [Numpy accessing elements in multi-dimensional array][numpy-accessing-elements-in-multi-dimensional-array]. This one is really weird and up till now no one gives an explanation for it.

But anyway the first technical blog; wow so exciting!


[python-reference-or-value]: http://stackoverflow.com/questions/8744113/python-list-by-value-not-by-reference
[numpy-accessing-elements-in-multi-dimensional-array]: http://stackoverflow.com/questions/40047621/numpy-muti-dimensional-array-index-is-out-of-bounds?noredirect=1#comment67372101_40047621