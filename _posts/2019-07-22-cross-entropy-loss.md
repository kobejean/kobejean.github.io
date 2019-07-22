---
layout:     post
title:      "The Benefits of Cross Entropy Loss"
subtitle:   "A look into the advantages of using cross entropy loss for classification problems."
date:       2019-07-22 00:02
author:     "Jean Atsumi Flaherty"
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

where $$\model$$ is the model output and.

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

## Computational Benefit

One thing you might ask is what difference would it make by using log probabilities instead of just the probabilities themselves? Well the main reason lies how much computational complexity you save by using logarithms.  As demonstrated in the section above, in order to compute the likelihood of the model, we need to calculate a joint probability over each dataset example. This involves multiplying all the per example probabilities together:

$$
\L(\theta | \D) = \prod_{(\x,\y) \in \D} P(\x,\y | \theta)
$$

Now if we were to do gradient decent on this joint probability directly, we would have to compute the derivative of this pi product. The problem with this however is that the number of terms of this derivative grows exponentially in the number of products, $$2^(n-1)$$ to be exact. This can get nasty very quickly.

However this issue can be avoided with log probabilities because by the product rule of logarithms, we can turn the pi product of probabilities inside the logarithm into a sum of logarithms:

$$
\log\left(\prod_{(\x,\y) \in \D} P(\x,\y | \theta)\right) = \sum_{(\x,\y) \in \D} \log(P(\x,\y | \theta))
$$

This way when we compute the gradient the number of terms in the derivative only grows linearly which saves us a lot of computation.

## Numerical stability

Another benefit of cross entropy losses/log probabilities is numerical stability. Since often times the joint probabilities get extremely tiny, we risk having issues with arithmetic underflow. Using log-probabilities solves this issue by keeping the values in a reasonable range.


## References

1. Morgan Giraud. *[ML notes: Why the log-likelihood?](https://blog.metaflow.fr/ml-notes-why-the-log-likelihood-24f7b6c40f83)*
2. Rob DiPietro. *[A Friendly Introduction to Cross-Entropy Loss](https://rdipietro.github.io/friendly-intro-to-cross-entropy-loss/#cross-entropy)*
