---
layout:     post
title:      "Batch Normalization (Part 1)"
subtitle:   "The myths and truth surrounding batch normalization."
date:       2019-04-10 22:00
author:     "Jean A. Flaherty"
header-img: "img/batch-norm/batch-norm-thumbnail-2.png"
category:   machine-learning
tags:       [machine learning]
---

Batch normalization is a widely used technique today in deep learning. The
[original paper](https://arxiv.org/abs/1502.03167)
took the machine learning community by storm when it was shown that batch
normalization could speed up training on ImageNet by factor of 14. This meant
that what would normally take two weeks to train could be achieved in a single
day with batch norm.

Although batch normalization has proven to be extremely effective and a
must-have in modern machine learning architectures, there are many
misconceptions about why batch normalization works so well. Some of the
original paper's theoretical claims have since been disproven and we are only
just beginning to understand the real reasons behind batch norm's effectiveness.

Here I will explain the initial explanation for batch norm's success,
as well as why it was wrong and what our current theories about batch norm are.
But before I get into that, let's take a look at what batch norm actually is.

## What is Batch Norm?

Batch normalization is a transformation that normalizes a given input to
have a standard deviation of $$\gamma$$ and a mean of $$\beta$$ using batch
statistics. $$\gamma$$ and $$\beta$$ are learned parameters. The formula for
batch norm looks like this:

$$
\begin{align}
    \hat{x}^{(k)} &= \frac{x^{(k)} - \mu}{\sqrt{\sigma^2 - \epsilon}} \\
    BN_{\gamma,\beta}(x^{(k)}) &= \gamma \hat{x}^{(k)} + \beta
\end{align}
$$

First the input is normalized by subtracting the batch sample mean, then
divided by the batch sample standard deviation (the $$\epsilon$$ is there to
avoid zero division). This makes $$\hat{x}$$ have zero mean and unit variance.
Then $$\hat{x}$$ is scaled by $$\gamma$$ and translated by $$\beta$$.

Here's the algorithm from the original paper:

![Batch Norm Algorithm]({{ site.baseurl }}/img/batch-norm/batch-norm-algorithm.png)

With a regular fully connected layer you will often see a formula written like
this:

$$
\begin{align}
    z &= Wu + b \\
    h &= \sigma(z)
\end{align}
$$

Where $$\sigma$$ is some non-linear function like $$ReLU$$, $$sigmoid$$,
$$softmax$$ etc. We call $$z$$ a layer's pre-activation and $$h$$ a layer's
activation. In the original paper, batch normalization is applied to the
pre-activation and the formula of a fully connected layer with batch norm looks
like this:

$$
\begin{align}
    z &= Wu \\
    h &= \sigma(BN_{\gamma,\beta}(z))
\end{align}
$$

Notice that we omit the bias $$b$$ in the pre-activation. This is because the
bias term doesn't get used since batch norm normalizes the pre-activations
to have $$\gamma$$ standard deviation and $$\gamma$$ mean. Our learned $$\beta$$
parameter serves as our bias.

## Internal Covariate Shift

In order to explain the initial motivation of the author's of batch norm, let
me explain a concept called Internal Covariate Shift or ICS for short.

![Internal Covariate Shift]({{ site.baseurl }}/img/batch-norm/internal-covariate-shift.png)

One problem we have when computing the gradient at each training step is that
our gradients only approximate the direction of steepest decent under the
assumption that each layer will receive the same distribution of inputs.
This is a false assumption because after we update our parameters, it's quite
likely that previous layers have changed so that a given layer is fed a slightly
different distribution of inputs. Internal Covariate Shift refers to this change
in activation distributions due to parameter updates during training. Because of
ICS, every time parameters are updated the model needs to learn a different set
of parameters to account for the change in network activation distributions.

If we could reduce ICS we would have more confidence in our gradients and we
could increase our learning rate. This was one of the motivations behind
batch norm. By using batch norm to fix the pre-activations/activations to have
$$\gamma$$ standard deviation and $$\beta$$ mean, the authors of the batch norm
paper thought it would help reduce ICS. As we will see in the next section, this
idea seems to have been proven wrong in a
[recent paper](https://arxiv.org/abs/1805.11604).

## ICS Doesn't Explain Batch Norm's Success

![Layer Distribution Chart Comparison]({{ site.baseurl }}/img/batch-norm/ics-myth-busted.png)

In a [recent paper](https://arxiv.org/abs/1805.11604) it was shown that batch
norm does not reduce internal covariate shift by much and sometimes even makes
it worse. In the paper they measured a broader notion of ICS by
comparing the gradients of layer parameters before and after updates to the
preceding layers. What they found was that despite the improvement in
performance, they found similar or even worse ICS in models with batch norm.

![ICS Chart Comparison]({{ site.baseurl }}/img/batch-norm/ics-myth-busted-2.png)

It turns out that since $$\gamma$$ and $$\beta$$ are learned parameters,
the network activation distributions still experience covariate shift.

So if ICS doesn't explain batch norm's success then what explains the 14x
train time speed up on ImageNet? What the recent paper revealed was that batch
norm has a smoothening effect on the loss landscape of neural networks measured
by the β-Lipschitzness of the loss function's gradient. A smaller Lipschitz
constant on the gradients means that our gradients are more predictable.
This gives us confidence that even if we take large steps with our computed
gradients, we won't veer far from where our true gradients should be taking us.
This allows us to increase the learning rate and also makes our path length
from initialization to global minimum much shorter because we avoid traveling
in random directions. Batch norm's effect on the smoothness of the loss
landscape seems to be the main reason for batch norm's success.

## References

1. Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. *arXiv preprint arXiv:1502.03167*, 2015.
2. Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry. How Does Batch Normalization Help Optimization?. *arXiv preprint arXiv:1805.11604*, 2018.
