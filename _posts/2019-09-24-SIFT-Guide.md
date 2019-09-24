---
layout: post
title:  "SIFT and Project 2: SIFTNet"
date:   2019-09-24 15:43:00 +0800
categories: jekyll update
---

Created by Richard Jiang on 24th, Sept, 2019. 

Having received a ton of questions on "what is going on?" and "how do I start?" during the office hours and on Piazza, I'll try explaining the basic concepts of SIFT here and also provide some guide on approaches to start building the network. Also, I'll try my best to incorporate some hints on how you can use PyTorch to achieve the goals. I decided to work on this tutorial just 10 min ago, so bear with me for some of the wordings :-)

## SIFT
All of this starts with the idea of image gradients: the layer you have implemented earlier and the **Ix** and **Iy** thing: these values tell you **how much the current pixel is changing relative to my neighbors**; the filtering using things like 
{% highlight Python %} 
[[1,0,-1],
 [2,0,-2],
 [1,0,-1]
] 
{% endhighlight %} 
is actualy telling you a **feature** about this pixel and the current image patch. So for each of the pixels in your image, there is this set of values: how much I'm changing in x, and y directions. 

Now consider a patch of pixels, say 8x8, then everyone of the 64 pixels has an Ix and Iy, and if we think about Ix and Iy as **vector magnitudes**, then we can say things like: for pixel a, it is mainly pointing towards the direction of Ix \times u + Iy \times v, where u and v represent the unit vector in x and y axis; what does this mean? How much does the gradient contribute to x and y directions, right? And the contribution is essentially the **projection onto the x and y axis** On the other hand, if our coordinate system consists of 8 directions/axis, we could do similar things like projecting the gradient vector onto these 8 directions and see **which direction this vector is contributing the most**. 

If we divide the 8x8 patch to 4 sub-regions (4x4 each), then for every sub-region, we should be able to tell one idea: on direction 1, how much contribution I received from this sub-region in total, on direction 2 how much contribution, etc ... and the end result is u0 = [v0, v1, v2, ..., v7]  (meaning for sub-region u0), and concatenating u0, u1, u2, u3, we have a thing called SIFT feature descriptor.

Why does this work as a feature? Well, let's consider the following mini-example (sorry for the bad drawing; here are 2 windows):

![window]({{richardejiang.github.io}}/assets/images/windows.jpeg){:id="windows"}

So the orange shape represents the window, and red squares mean the 4 sub-regions, and suppose the blue vectors give the most contributing directions of this patch. Then if we concatenate the 4 vectors together from the left image, we can tell that for a window, we end up with a descriptor where the first region direction is pointing to top-left, second pointing to top-right, etc... Let's say we take an image of the window from another angle (the image on the right), although the directions may differ a little. bit, but the concatenated vectors are similar right?

## SIFT Orientation
Ok so get down to impl. We say we have 8 directions, and we want to find out how the gradient vector is contributing to each of them. How to do it? Well as the comment suggests, "projection". Essentially you are projecting the vector onto each of the direction vectors, and consider first: how to get the 8 direction vectors? 

![window]({{richardejiang.github.io}}/assets/images/directions.jpeg){:id="windows"}

We want to span the directions uniformlly, so you know the angle theta, then how do you represent the vector? Simple right?

Ok so why do we need this? Remember we want to find the **projection** onto each of these direction vectors, and a projection is relevant to inner product right?

![window]({{richardejiang.github.io}}/assets/images/inner.jpeg){:id="windows"}

So now you have the basic idea on the concepts. In the code for this part, since it's an inner product, where it's essentially **element-wise multiplication and summation**. Does this sound familiar? You will find nn.Conv2d useful here, and remember we want to project the gradient vector onto directions, so you may convolve xxx with xxx... (I'll leave the rest to you otherwise it'll be like giving away the answer.)

## Histogram
So what we want is calculating the contribution to each of the sub-regions (the figure on the Szelinski book is clearer for the idea behind this, refer to Figure 4.18 on page 224). What have done so far? Every pixel there are 8 values for 8 directions, and we want to calculate **which direction is this pixel contributing the most**? And the original gradient value of the pixel will be marked for the magnitude of its direction vector. Hence the most straightforward logic is to loop, loop, and loop:
{% highlight Python %} 
for row:
    for column:
        for each of the 8 directions:
            if this is the direction I'm contributing the most: 
                mark the magnitude
{% endhighlight %} 

But as you can image this will take forever to run; that's why you can find in the comment section that this kind of impl will be penalized. So think about how to speed it up. A quick hint: in np you can do things like:
{% highlight Python %} 
a = np.array([1,2,3,4,5])
idx = np.array([0,2])
print(a[idx]) # this will give you [1, 3]
{% endhighlight %} 

