---
layout:     post
title:      "Batch Normalization (Part 2)"
subtitle:   "The smoothening effect of batch normalization."
date:       2019-04-28 14:45
author:     "Jean A. Flaherty"
header-img: "img/batch-norm/beta-smoothness.png"
category:   machine-learning
tags:       [machine learning]
---
<div style="display: none">
<!-- LaTeX Helpers -->
$$
\newcommand{\vect}[1]{ \boldsymbol{#1} }
\newcommand{\batch}[1]{ ^{({#1})} }
\newcommand{\grad}[1]{ \nabla#1 }
\newcommand{\gradWrt}[2]{ \nabla_{#2}#1 }
\newcommand{\gradDir}[1]{ \frac{ \grad{#1} }{ \| \grad{#1} \|} }
\newcommand{\gradDirWrt}[2]{ \frac{ \gradWrt{#1}{#2} }{\| \gradWrt{#1}{#2} \|} }
\newcommand{\partialD}[2]{ \frac{ \partial#1 }{ \partial#2 } }
\newcommand{\partialDTwo}[3]{ \frac{ \partial#1 }{ \partial#2\partial#3 } }

\newcommand{\L}{ \mathcal{L} }
\newcommand{\B}{ \mathbb{B} }
\newcommand{\X}{ \boldsymbol{X} }
\newcommand{\H}{ \boldsymbol{H} }
\newcommand{\y}{ \vect{y} }
\newcommand{\x}{ \vect{x} }
\newcommand{\g}{ \vect{g} }
\newcommand{\fbeta}{ \beta^* }
$$
</div>

[Previously]({% post_url 2019-04-10-batch-normalization-part-1 %}), we went over
how batch norm works, the concept of internal covariate shift, and the real reason for batch norm's success. Continuing the topic of batch norm, we will take a deeper look into
the smoothening effect of batch norm and why it helps us train our models faster.

## *β*-smoothness
In the paper [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604),
the authors suggested that the real explanation for batch norm's success had little
to do with internal covariate shift and more to do with the smoothness of the loss landscape.
They measured smoothness by *β*-smoothness which is the Lipschitz constant of the
gradient. A function is considered *β*-smooth if it satisfies the following condition.

$$ \forall \x,\y \in \R^n : \| \nabla f(\x) - \nabla f(\y) \| \leq \beta \| \x - \y \| $$

And the measure of *β*-smoothness is defined as the smallest *β* constant to
satisfy this condition. If your function is twice differentiable then you can
derive *β* in terms of the hessian by the mean value theorem.

$$
\begin{align}
  \beta &= \max_{ \x,\y\in\R^n }
  \frac{
    \| \grad{ f(\x) } - \grad{ f(\y) } \|
  }{
    \| \x - \y \|
  }\\
  &= \max_{ \x\in\R^n } \left\Vert \partialDTwo{f}{\x}{\x} \right\Vert
\end{align}
$$

An important caveat in our case is that the *β*-smoothness is not bounded in a
global sense because of the non-linearities in a neural network. The authors
were actually measuring the *effective* *β*-smoothness, that is the *β*-smoothness
we observe as we move along the gradient.

$$
\fbeta_t = \max_{ 1 \leq b \leq n } \frac{
  \| \gradWrt{\L_t}{\y_t}\batch{b} - \gradWrt{\L_{t+1}}{\y_{t+1}}\batch{b} \|
}{
  \| \y_t\batch{b} - \y_{t+1}\batch{b} \|
}
$$

where $$\L_t$$ is our loss at time-step *t*, and *n* is the size of the
mini-batch.

![β-Smoothness]({{ site.baseurl }}/img/batch-norm/beta-smoothness.png)

The figure above compares the "effective" *β*-smoothness at each training step
between a model with and without batch norm.

## Theoretical Analysis

Beyond just empirical evidence of improved *β*-smoothness under batch norm,
the paper also provides theoretical analysis on the smoothening effect.
To see how batch norm improves *β*-smoothness, the authors compared the
quadratic form of the loss Hessian w.r.t the pre-activations in the direction
of the gradient.

$$
\left( \gradWrt{\L}{\y_j} \right)^\intercal
\partialDTwo{\L}{\y_j}{\y_j}
\left( \gradWrt{\L}{\y_j} \right)
$$

This term captures the effective *β*-smoothness constant of the loss landscape.
When this term is lower in value, we improve our effective *β*-smoothness.
Let's define $$\L$$ to be the loss of a model *without* batch norm and $$\hat{\L}$$
to be the loss of the same model but *with* batch norm. For convenience let's
also define
$$\hat\y = \frac{\y - \mu}{\sigma}$$,
$$\g_j = \gradWrt{\L}{\y_j}$$, and
$$\H_{j,j} = \partialDTwo{\L}{\y_j}{\y_j}$$.
In [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604), the following relationship was proven:

$$
\left( \gradWrt{\hat\L}{\y_j} \right)^\intercal
\partialDTwo{\hat\L}{\y_j}{\y_j}
\left( \gradWrt{\hat\L}{\y_j} \right)

\leq

\frac{\gamma^2}{\sigma^2}
\left( \gradWrt{\hat\L}{\y_j} \right)^\intercal
\H_{j,j}
\left( \gradWrt{\hat\L}{\y_j} \right)
-
\frac{\gamma}{m\sigma^2}
\left<\g_j,\hat{\y}_j\right>
\left\Vert \partialD{\hat\L}{\y_j} \right\Vert^2
$$  

and if we also have that $$\H_{j,j}$$ preserves the relative norms of
$$\g$$ and $$\gradWrt{\hat\L}{\y_j}$$,

$$
\left( \gradWrt{\hat\L}{\y_j} \right)^\intercal
\partialDTwo{\hat\L}{\y_j}{\y_j}
\left( \gradWrt{\hat\L}{\y_j} \right)

\leq

\frac{\gamma^2}{\sigma^2}
\left(
  \g_j^\intercal
  \H_{j,j}
  \g_j
  -
  \frac{1}{m\gamma}
  \left<\g_j,\hat{\y}_j\right>
  \left\Vert \partialD{\hat\L}{\y_j} \right\Vert^2
\right)
$$

The division by $$\sigma^2$$ on the right side of the equation shows that our gradients are are more predictable and robust to pre-activation variance when adding batch norm. Overall, as the paper notes, when the loss Hessian w.r.t the pre-activations are positive semi-definite which is true for most models, and the negative gradient points toward the loss minimum (when $$ \left<\g_j,\hat{\y}_j\right> \geq 0 $$) then batch norm has more predictive gradient steps.

## How does smoothness improve training time?

There's some work in theoretical convex optimization showing how *β*-smoothness improves the convergence rate of convex *β*-smooth functions. For more details I recommend reading [Gradient descent for smooth and for strongly convex functions](http://mitliagkas.github.io/ift6085/ift-6085-lecture-3-notes.pdf) and section 1.2 of [Introductory Lectures on Convex Programming](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.693.855&rep=rep1&type=pdf) but the fundamental take away is that for convex optimization of *β*-smooth functions, the optimal learning rate is around $$\frac{1}{\beta}$$. And the difference between the global optimum value and the t-th training step value (with $$\frac{1}{\beta}$$ as our learning rate) is:

$$
f(\x_t) - f(\x^*) \leq \frac{2\beta\|\x_1 - \x^*\|}{t - 1}
$$

This means that if we can halve the *β* constant with all else equal, there's a good chance we can afford to double the learning rate and we can halve the distance between the current loss and the optimal loss in the same number of time-steps.

It's important to consider that simply re-scaling the parameters in a model, though will reduce your *β* constant, won't give you improvements in convergence rate. What you really care about is more or less the ratio between path length from initialization to global minimum and the optimal learning rate. If you simply re-scale your model parameters, your optimal learning rate will inevitably be scaled by the same factor, canceling out any reduction in path length when considering convergence rate. The authors of [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) did not forget to consider this and pointed out that the improvement in *β*-smoothness still holds when we simply re-parametrize a non-batch norm model to a batch norm model without changing its effective scaling. In addition, they prove that batch norm improves initialization in the following way:

$$
\left\Vert W_0 - \hat{W}^* \right\Vert^2

\leq

\left\Vert W_0 - W^* \right\Vert^2
-
\frac{1}{\| W^* \|^2}
\left(
  \| W^* \|^2
  -
  \left<
    W^* ,
    W_0
  \right>
\right)
$$

Where $$W_0$$ is your initial weights, $$\hat{W}^* $$ is your closest optimum under batch norm, and $$W^* $$ is your closest optimum without batch norm.


## References

1. Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, and Aleksander Madry. How Does Batch Normalization Help Optimization?. *arXiv preprint arXiv:1805.11604*, 2018.
2. Yurii Nesterov. *Introductory Lectures on Convex Optimization: A Basic Course*. Springer Science & Business Media, 2014.
3. Ioannis Mitliagkas. *IFT 6085 - Lecture 3: Gradient descent for smooth and for strongly convex functions*. IFT 6085, University of Montreal, 2018.
