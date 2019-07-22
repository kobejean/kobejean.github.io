---
layout:     post
title:      "The Benefits of Cross Entropy Loss"
subtitle:   "A look into the advantages of using cross entropy loss for classification problems."
date:       2019-07-21 14:45
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

where $$\D$$ is our dataset and $$ \theta $$ is our model parameters.

Since we are usually assuming that our data is independent and identically distributed, the conditional probability can be rewritten as a join probability over each example:

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

where $$\model_i$$ is the component of the correct category in the model output.

Notice the cross entropy of the output vector is equal to the log of $$\model_i$$ because our "true" distribution is a one hot vector:

$$
\begin{align}
  H(\y, f(\x)) &= - \sum_{j=1}^{n} \y_j\log(\model_j) \\
  &= - \log(\model_i) \\
\end{align}
$$

where $$\y$$ is one hot encoded the target vector.

So in total we have:

$$
\begin{align}
    \argmax_\theta \L(\theta | \D) &= \argmax_\theta P(\D | \theta) \\
    &= \argmax_\theta \prod_{(\x,\y) \in \D} P(\x,\y | \theta) \\
    &= \argmin_\theta \sum_{(\x,\y) \in \D} -\log(P(\x,\y | \theta)) \\
    &= \argmin_\theta \sum_{(\x,\y) \in \D} -\log(\model_i) \\
    &= \argmin_\theta \sum_{(\x,\y) \in \D} -\sum_{j=1}^{n} \y_j\log(\model_j) \\
    &= \argmin_\theta \sum_{(\x,\y) \in \D} H(\y, \model) \\
\end{align}
$$

Thus we have shown that maximizing the likelihood of a classification model is equivalent to minimizing the cross entropy of the models categorical output vector and thus cross entropy loss has a valid theoretical justification.

## Computational Benefit




## References

1. Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry. How Does Batch Normalization Help Optimization?. *arXiv preprint arXiv:1805.11604*, 2018.
2. Yurii Nesterov. *Introductory Lectures on Convex Optimization: A Basic Course*. Springer Science & Business Media, 2014.
3. Ioannis Mitliagkas. *IFT 6085 - Lecture 3: Gradient descent for smooth and for strongly convex functions*. IFT 6085, University of Montreal, 2018.
