---
layout: post
title:  "Okay Python again: when people forget about the data types"
date:   2017-08-29 16:53:00 +0800
categories: jekyll update
---

Created by Richard Jiang on 29th, Aug, 2017. 

Okay haven't seen an update so long, and yet it shall continue. The research assistant job goes ... dunno, maybe good, maybe bad; the upcoming GRE test appears to be the biggest problem and keeps haunting me for the past weeks. Vocab so difficult!

Nevertheless, let's look at the new peoblem. It has been advised in various ways that when a person just starts to learn about programming, start with C, C++, or Java; Python may not be a suitable choice. And over the past two years I gradually realize how witty this is: in a nutshell, Python is just too 'easy'. There is no data type declaration, and this causes confusion sometimes. Consider the code below:

{% highlight Python %}
import numpy as np

a=np.array([5,6,7,8,9])
b=np.array([5,6,7,8,9])

scoreA = np.array([float(1) / (i + 1) for i in range(len(a))])
scoreB = np.array([0 for i in range(len(b))])

for eleA in a:
    if eleA in b:
        i, = np.where(b == eleA)
        i = i[0]
        j, = np.where(a == eleA)
        j = j[0]
        scoreB[i] = scoreA[j]

        print "B is: %f" % scoreB[i]
        print "A is: %f" % scoreA[j]
{% endhighlight %}

So the idea is: for numpy array a and b, they have a score list, and when an element from a is found in b, the score from that element in a should be copied to b. But the output is as belows:

{% highlight Python %}
B is: 1.000000
A is: 1.000000
B is: 0.000000
A is: 0.500000
B is: 0.000000
A is: 0.333333
B is: 0.000000
A is: 0.250000
B is: 0.000000
A is: 0.200000
{% endhighlight %}

So the copy-paste is not done. And the problem simply lies here: change the declaration of scoreB to:

{% highlight Python %}
scoreB = np.array([0 for i in range(len(b))],dtype=float)
{% endhighlight %}

As Int(a float number < 1) will yield 0.

Yeah I agree this is a stupid problem, but when you are used to Python, sometimes the concept of data type and how it may affect the code just becomes trivial, and you are no longer sensitive to them. Therefore, code is cool, and Python has prevailed thanks to its superior handiness and simplicity (and, of course, the available libs), but don't forget about the initial problems, though.

PS: sure, a cleaner way to write this logic would be: 
{% highlight Python %}
mask = a==b; 
scoreB[mask] = scoreA[mask]
{% endhighlight %}
