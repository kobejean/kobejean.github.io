---
layout:     post
title:      "The Benefits of Cross Entropy Loss"
subtitle:   "A look into the advantages of using cross entropy loss for classification problems."
date:       2019-07-22 00:02
author:     "Jean A. Flaherty"
header-img: "img/cross-entropy/cross-entropy-thumbnail.png"
category:   machine-learning
tags:       [machine learning]
---
<div style="display: none">
<!-- LaTeX Helpers -->
$$
\newcommand{\argmax}{\mathop{\mathrm{argmax}}}
\newcommand{\argmin}{\mathop{\mathrm{argmin}}}
\newcommand{\vect}[1]{ \boldsymbol{#1} }
\newcommand{\batch}[1]{ ^{({#1})} }
\newcommand{\grad}[1]{ \nabla#1 }
\newcommand{\gradWrt}[2]{ \nabla_{#2}#1 }
\newcommand{\gradDir}[1]{ \frac{ \grad{#1} }{ \| \grad{#1} \|} }
\newcommand{\gradDirWrt}[2]{ \frac{ \gradWrt{#1}{#2} }{\| \gradWrt{#1}{#2} \|} }
\newcommand{\partialD}[2]{ \frac{ \partial#1 }{ \partial#2 } }
\newcommand{\partialDTwo}[3]{ \frac{ \partial#1 }{ \partial#2\partial#3 } }
\newcommand{\derivativeWrt}[2]{ \frac{ d#1 }{ d#2 } }

\newcommand{\L}{ \mathcal{L} }
\newcommand{\P}{ P }
\newcommand{\D}{ D }
\newcommand{\R}{ \mathbb{R} }
\newcommand{\H}{ \boldsymbol{H} }
\newcommand{\y}{ \vect{y} }\hat{x}^{(k)}
\newcommand{\x}{ \vect{x} }
\newcommand{\model}{ f(\x,\theta) }
$$
</div>

Cross entropy loss is almost always used for classification problems in machine learning.  I thought it would be interesting to look into the theory and reasoning behind it's wide usage.  Not as much as I expected was written on the subject, but from what little I could find I learned a few interesting things.  This post will be more about explaining the justification and benefits of cross entropy loss rather than explaining what cross entropy actually is.  Therefore, if you don't know what cross entropy is, there are many great sources on the internet that will explain it much better than I ever could so please learn about cross entropy before continuing.

## Theoretical Justification

It's important to know why cross entropy makes sense as a loss function. Under the framework of maximum likelihood estimation, the goal of machine learning is to maximize the likelihood of our parameters given our data, which is equivalent to the probability of our data given our parameters:

$$
\L(\theta | \D) = \P(\D | \theta)
$$

where $$\D$$ is our dataset (a set of pairs of input and target vectors $$\x$$ and $$\y$$) and $$ \theta $$ is our model parameters.

Since the dataset $$\D$$ has multiple datum, the conditional probability can be rewritten as a joint probability of per example probabilities. Note that we are assuming that our data is independent and identically distributed. This assumption allows us to compute the joint probability by simply multiplying the per example conditional probabilities:

$$
P(\D | \theta) \stackrel{i.i.d}{=} \prod_{(\x,\y) \in \D} P(\x, \y | \theta)
$$

And since the logarithm is a monotonic function, maximizing the likelihood is equivalent to minimizing the negative log-likelihood of our parameters given our data.

$$
\argmax_\theta \L(\theta | \D) = \argmin_\theta (-\log(P(\D | \theta)))
$$

In classification models, often the output vector is interpreted as a categorical probability distribution and thus we have:

$$
P(\x,\y | \theta) = \model_i
$$

where $$\model$$ is the model output and $$i$$ is the index of the correct category.

Notice the cross entropy of the output vector is equal to $$-\log(\model_i)$$ because our "true" distribution is a one hot vector:

$$
\begin{align}
  H(\y, \model) &= - \sum_{j=1}^{n} \y_j\log(\model_j) \\
  &= - \y_i\log(\model_i) \\
  &= - \log(\model_i) \\
\end{align}
$$

where $$\y$$ is the one hot encoded target vector.

So in total we have:

$$
\begin{align}
    \argmax_\theta \L(\theta | \D) &= \argmax_\theta P(\D | \theta) \\
    &= \argmax_\theta \prod_{(\x,\y) \in \D} P(\x,\y | \theta) \\
    &= \argmin_\theta \sum_{(\x,\y) \in \D} -\log(P(\x,\y | \theta)) \\
    &= \argmin_\theta \sum_{(\x,\y) \in \D} -\log(\model_i) \textrm{, where } i \textrm{ is the index of the correct category } \\
    &= \argmin_\theta \sum_{(\x,\y) \in \D} -\sum_{j=1}^{n} \y_j\log(\model_j) \\
    &= \argmin_\theta \sum_{(\x,\y) \in \D} H(\y, \model) \\
\end{align}
$$

Thus we have shown that maximizing the likelihood of a classification model is equivalent to minimizing the cross entropy of the models categorical output vector and thus cross entropy loss has a valid theoretical justification.

## Numerical Stability

One thing you might ask is what difference would it make by using log probabilities instead of just the probabilities themselves given the logarithm is monotonic? Well one of the main reasons lies in its property of numerical stability.  As demonstrated in the section above, in order to compute the likelihood of the model, we need to calculate a joint probability over each dataset example. This involves multiplying all the per example probabilities together:

$$
\L(\theta | \D) = \prod_{(\x,\y) \in \D} P(\x,\y | \theta)
$$

This pi product becomes very tiny. Consider the case where each probability is around 0.01 (e.g trying to predict a class over 100 classes) and we are using a batch size of 128. The joint probability would be around $$1 \times 10^{-256}$$ which is definitely low enough to cause arithmetic underflow.

However this issue can be avoided with log probabilities because by the product rule of logarithms, we can turn the pi product of probabilities inside the logarithm into a sum of logarithms:

$$
\log\left(\prod_{(\x,\y) \in \D} P(\x,\y | \theta)\right) = \sum_{(\x,\y) \in \D} \log(P(\x,\y | \theta))
$$

Using log-probabilities keeps the values in a reasonable range. It also keeps computing the gradient simple because it is easier to aggregate the gradient of a sum of functions than it is a product of functions.

## Well-behaved Gradients

Using log-probabilities has the additional effect of keeping gradients from varying too widely. Many probability distributions we deal with in machine learning belong to the exponential family. Take for example a normal distribution:

$$
p(x) = e^{-\frac{1}{2} x^2}
$$

Notice what happens when we turn this into a negative log-probability and take the derivative:

$$
\begin{align}
  \derivativeWrt{(-\log(p(x)))}{x} &= \derivativeWrt{(-log(e^{-\frac{1}{2} x^2}))}{x} \\
  &= \derivativeWrt{(\frac{1}{2} x^2)}{x} \\
  &= x \\
\end{align}
$$

Notice the derivative $$x$$ will give us exactly the right $$x$$ value ($$0$$) after the update rule to maximize the likelihood if our learning rate is set to 1.

$$
\begin{align}
  x' &= x - \eta \derivativeWrt{(-\log(p(x)))}{x} \\
  &= x - \eta x \\
  &= 0 \\
  &= \argmax_x p(x) \\
\end{align}
$$

Obviously this is just a toy example but other probability distributions of the exponential family will also do quite well as the log probability is at most polynomial in the parameters. This keeps the gradient from varying widely and means that the same learning rate should give us consistent step sizes.

Also consider the common case of using softmax before the cross entropy loss:

$$
f(\x) = \frac{e^{\x}}{\sum_{i=0}^{n}e^{\x_i}}
$$

Recall that the cross entropy loss is given by:

$$
H(y, f(\x)) = -\y\cdot\log(f(\x))
$$

Now try compute the gradient of the cross entropy loss w.r.t $$\x$$.

$$
\begin{align}
  \partialD{H(\y, f(\x))}{\x} &= \partialD{\left( -\y\cdot\log(f(\x)) \right)}{\x} \\
  &= -\y^\intercal \partialD{\log\left(\frac{e^\x}{\sum_{i=0}^{n}e^{\x_i}}\right)}{\x} \\
  &= -\y^\intercal \partialD{\left( \log(e^\x) - \vect{1}\log\left(\sum_{i=1}^{n}e^{\x_i}\right) \right)}{\x} \\
  &= -\y^\intercal \left( I - \vect{1}\partialD{\log\left(\sum_{i=1}^{n}e^{\x_i}\right)}{\x} \right) \\
  &= -\y^\intercal \left( I - \vect{1} \left(\frac{e^\x}{\sum_{i=1}^{n}e^{\x_i}} \right)^\intercal \right)  \\
  &= -\y^\intercal \left( I - \vect{1}f(\x)^\intercal \right)  \\
  &= \left(f(\x)\vect{1}^\intercal - I \right)\y \\
  &= f(\x)\vect{1}^\intercal\y - \y \\
  &= f(\x)\sum_{i=1}^{n}\y_i - \y \\
  &= f(\x) - \y \\
\end{align}
$$

As we can see this gives us a well-behaved gradient that is bounded by $$[-1,1]$$ for each component.


## References

1. Morgan Giraud. *[ML notes: Why the log-likelihood?](https://blog.metaflow.fr/ml-notes-why-the-log-likelihood-24f7b6c40f83)*
2. Rob DiPietro. *[A Friendly Introduction to Cross-Entropy Loss](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/#cross-entropy)*
