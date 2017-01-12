---
layout: post
title:  "Swift Optionals: Elegant, but be Careful"
date:   2017-01-12 16:53:00 +0800
categories: jekyll update
---

Created by Richard Jiang on 12th, Jan, 2017. 

New semester just started, and I decided to take CS3217 (possibly the most, if not one of, tiring, difficult, yet sounds-to-be-cool-and-fulfilling module in SoC). It's my last semester here as an undergraduate, so I decided to challenge myself by follow this painful path.

Nevertheless, assignment started last week. Entitled as 'Advanced Software Engineering Concepts', the module will, in my opinion, emphasizes a lot on the coding style and cleanness. Therefore, after finishing it, I went through some of the confusing part again today. I believe for most of the beginners to Swift, the concept of 'Optionals' is, well, not so friendly. So is to me. Look at the code snippet below:

{% highlight Swift %}
var arrayForm = [T]()
var currNode = self.head
while (currNode != nil) {
    arrayForm.append((currNode?.nodeContent)!)
    currNode = currNode?.nextNode
}
{% endhighlight %}

It's a simple assignment, requiring to implement Queue and Stack with some fundamental methods. Above is part of the code of toArray() method. As 
{% highlight Swift %}
if let currNode = self.head {
	// do sth
}
{% endhighlight %}
is the standard way of utilizing optional chaining and reduce the tediousness of writing {} again and again, I converted the above logic to sth like this:

{% highlight Swift %}
var arrayForm = [T]()
if let currNode = self.head {
	arrayForm.append(currNode.nodeContent)
	while let currNode = currNode.nextNode {
		arrayForm.append(currNode.nodeContent)
	}
}
{% endhighlight %}

And then, test cases failed as expected. It can be told that here, 'if let' and 'while let' statements are both quite elegant and demonstrate the purpose and powerfulness of Swift, especially the essence of Optionals, which is to keep it SAFE. However, the new thing about Swift is, 'let' and 'var' are different, yet for someone like me who has been used to Python where there is no things like 'private static final', sometimes we do forget it (yeah I have been working using Python for the past year and no other language). Once you created sth and declared it as constant, don't mutate it. Well, you cannot, as in my case the test cases just hang there.

So I guess one lesson from this very first asignment is, learning Swift style is good, but there is no necessity to do it completely. '!= nil' did nothing wrong and why abandon it?