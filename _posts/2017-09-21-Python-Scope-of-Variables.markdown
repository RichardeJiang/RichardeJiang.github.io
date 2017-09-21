---
layout: post
title:  "Python Note: Scope of Variables"
date:   2017-09-21 14:33:00 +0800
categories: jekyll update
---

Created by Richard Jiang on 21st, Sept, 2017. 

Accidentally captured this point from a seeming bug.

All starts from the XML file parsing task. Currently what I have in hand is the collection of the ACM Periodical Data (which means all coneferences papers, etc), in XML format, and my main text mining job launched from here. As some of the XML files are relatively out-dated, the format might not be rigorous enough for the parser (I'm using the *minidom* lib) to understand, so I got this for processing:

{% highlight Python %}
def XmlParsing(targetFile, targetTag):
	try:
		DOMTree = xml.dom.minidom.parse(targetFile)
	except xml.parsers.expat.ExpatError, e:
		print "The file causing the error is: ", fileName
		print "The detailed error is: %s" %e
	else:
		collection = DOMTree.documentElement

		resultList = collection.getElementsByTagName(targetTag)
		return resultList

	return "ERROR"

{% endhighlight %}

I'm not sure anyone has spotted the problem or not; nonetheless, let's check out this line:

`print "The file causing the error is: ", fileName`

Where is this `fileName`?

I spotted the error as from the printed out messages, it seems that only one file is problematic, and the program hangs there. I even tried removing that particular file, but soon noticed that another file is behaving the same way! Eventually I came to this `fileName`.

But shouldn't it be showing compilation error? Why is the code able to run without any error?

So, let's consider the snippet below:

{% highlight Python %}
def write():
	print "the file name is %s" % fileName
	return

if __name__ == "__main__":
	print "We define fileName first"
	ele = "acc1002.xml"
	a=["firstOne", "secondOne", "thirdOne"]
	for fileName in a:
	        ele = fileName
	print "Now in the main function, calling write()"
	write()

{% endhighlight %}

If the above reasoning holds, then it can be inferred that the output is:

`the file name is thirdOne`

And indeed it is.

When we are accustomed to Python's indentation style, it's just intuitive to take the code with the same number of tabs (okay spaces are also there, you win) as the same 'level', so 'why can a variable like fileName be held outside the main function?'

And there is just no such `main` function.

The above code is same when you write sth like this:

{% highlight Python %}
def write():
	print "the file name is %s" % fileName
	return


print "We define fileName first"
ele = "acc1002.xml"
...

{% endhighlight %}

And therefore, the lesson to take here is: in Python script, the code outside all the functions is the top level, and you are actually declaring variables which are global. Then of course they can be used inside those functions of lower hierarchies.
