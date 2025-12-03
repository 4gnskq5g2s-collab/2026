---
layout: distill
title: Discretisation invariance
description:
  We are going to talk about discretisation invariance - a recent innovation in scientific machine learning. Discretisation invariance is a requirement that ensures the architecture can process inputs of different resolutions. We will formally define this property, provide examples, generate datasets, train architectures, and discuss whether discretisation invariance is living up to its promise.
date: 2026-04-27
future: true
htmlwidgets: true
hidden: true

authors:
  - name: Anonymous

bibliography: 2026-04-27-discretisation-invariance.bib

toc:
  - name: Introduction and motivation
  - name: What is discretisation invariance?
  - name: Examples and non-examples
    subsections:
      - name: Fourier Neural Operator
      - name: Convolutional neural networks
      - name: Deep Operator Network with linear observations
      - name: PCA-Net
      - name: Neural fields
      - name: Graph neural networks
      - name: Graph kernel networks
  - name: Training and evaluation
  - name: The role of discretisation invariance

---
## Introduction and motivation
Partial differential equations (PDEs) are the gold standard in scientific modelling. With rare exceptions PDEs are solved numerically, and the goal is always to come up with a reliable, accurate algorithm that delivers a solution as cheap as possible. Neural networks were tried as a solution method starting at least from the 1990s <d-cite key="lee1990neural"></d-cite>, <d-cite key="lagaris1998artificial"></d-cite>. Over time, methods based on neural networks converged to two dominant settings: physics-informed neural networks (PINNs, <d-cite key="lagaris1998artificial"></d-cite>, <d-cite key="raissi2017physics"></d-cite>) and operator learning <d-cite key="kovachki2023neural"></d-cite>, <d-cite key="lu2021learning"></d-cite>. PINNs is an unsupervised techniques that directly aims to solve PDE. Operator learning, the approach we consider here, is a supervised technique aiming to amortize the cost of parametric PDE solution in the "multi-query setting". We explain the setup of operator learning below.

We start by specifying a model of interest in the form of PDE. For the sake of example we consider stationary diffusion equation

$$
\begin{aligned}
-&\frac{\partial}{\partial x}\left(k(x, y)\frac{\partial}{\partial x} u(x, y)\right)-\frac{\partial}{\partial y}\left(k(x, y)\frac{\partial}{\partial y} u(x, y)\right) = f(x, y),\\
&x,y\in\Gamma=(0, 1)^2,\,u(x, 0) = u(x, 1) = u(0, y) = u(1, y) = 0.
\end{aligned}
$$

This PDE naturally appears in modelling of multiphase fluid flow, heat conduction and electrostatic problems in heterogeneous media.

PDE contains two parameters: diffusivity coefficient $k(x, y) > 0$ and the source term $f(x, y)$. We assume that we need to solve the stationary diffusion equation repeatedly for a large set of parameters drawn from a joint probability distribution $k(x, y), f(x, y) \sim p_{f, k}$. One may simply call a classical solver for each new pair of parameters, but it can be more advantageous to exploit information recovered from already obtained solutions.

This can be done in a standard regression framework: collect dataset $\left(f_1, k_1, u_1\right), \dots, \left(f_{M}, k_{M}, u_{M}\right)$, select parametric model $\mathcal{N}_{\theta}$ and train it with $L_2$ loss function

$$
\theta^{\star} = \arg\min_{\theta}\mathbb{E}_{k, f\sim p_{k, f}}\left\|\mathcal{N}_{\theta}(k, f) - u\right\|_2^2 \simeq \arg\min_{\theta} \frac{1}{N}\sum_{i=1}^{N}\left\|\mathcal{N}_{\theta}(k_i, f_i) - u_i\right\|_2^2.
$$

All standard techniques of machine learning apply: cross-validation, gradient descent methods, regularisation, etc.

When PDE is discretised, continuous physical fields $k(x, y), f(x, y), u(x, y)$ become matrices, e.g., $k_{ij} = k(x_i, y_j)$ where $(x_i, y_j)$ is a point on regular grid. In this case, the regression problem is conceptually similar to classical image to image tasks: segmentation, denoising, superresolution, etc.

Recently, a group of researchers suggested that regression problem for PDE is more than learning of image to image map <d-cite key="li2020neural"></d-cite>, <d-cite key="li2020fourier"></d-cite>. They argued that primal objects are functions themselves, not a particular way they are summarised with finite data. For example, one may represent function $k(x, y)$ on the grid with $N\times N$ points, or on the grid with $2N\times 2N$ points, or as a set of coefficients $c_{ij}$ in a finite series $k(x, y) = \sum_{i, j=1}^{N} c_{ij}\phi_{i}(x)\phi_j(y)$. Particular representations are different, but the function $k(x, y)$ remains the same in all cases. Is it possible to build a neural network that is, to a degree, agnostic to the choice of particular discretisation? The answer is positive, and the architectures with such property are now called *discretisation invariant* or *discretisation agnostic*.

In this note we address several questions about discretisation invariance architectures:
1. What is discretisation invariance? How to define it formally?
2. How discretisation invariant architectures are built?
3. Why is discretisation invariance important?

## What is discretisation invariance?
Intuitively, discretisation invariant architectures consistently map functions for different resolutions: when more details appear in the input, we expect to see more details in the output.

{% include figure.liquid path="assets/img/2026-04-27-discretisation-invariance/waves.png" class="img-fluid" %}
<div class="caption">
    An example of discretisation invariant map $\psi = \mathcal{F}(\phi)$. When input $\phi$ is available on the refined grid the output $\psi$ is refined too.
</div>

To slightly formalise the illustration above we define sampling and interpolation operators.

Sampling operator $\mathcal{S}\_{N}:\mathcal{C}\_{[0, 1]}\rightarrow \mathbb{R}^{N}$ takes function $f$ from the space of continuous functions $\mathcal{C}\_{[0, 1]}$, and outputs its values on the uniform grid $x_{i} = i / (N-1),\,i=0,\dots,(N-1)$.

Interpolation operator $\mathcal{I}\_{N}:\mathbb{R}^{N}\rightarrow \mathcal{C}\_{[0, 1]}$ performs an inverse operation: from a set of samples $f(x\_i)$ it reconstructs function $\widetilde{f}$ by linear interpolation $\widetilde{f}(x) = \left(f(x\_{i}) (x\_{i+1} - x) + f(x\_{i+1}) (x - x\_{i})\right) / (x\_{i+1} - x\_{i})$ for $x \in[x\_{i}, x\_{i+1}]$.

In general $f(x) \neq \mathcal{I}\_{N}\left(\mathcal{S}\_{N}(f)\right)(x)$ but as $N$ grows, composition $\mathcal{I}\_{N}\mathcal{S}_{N}$ becomes closer to identity in the standard $L_2$ norm: $\lim\limits\_{k\rightarrow\infty}\left\|f - \mathcal{I}\_{k}\mathcal{S}\_{k} f\right\|\_2 = 0$.

Having sampling $\mathcal{S}$ and interpolation $\mathcal{I}$ operators, we call map $F\_{k}:\mathbb{R}^{k}\rightarrow \mathbb{R}^{k}$ discretisation invariant if sequence $\psi_k=\mathcal{I}\_{k}\left(F_{k}\left(\mathcal{S}\_{k}(\phi)\right)\right)$ converges to a unique element of space $\mathcal{C}\_{[0, 1]}$ for each input $\phi$.

Our definition requires several clarifications:
1. We choose particular operators $\mathcal{S}\_{k}$, $\mathcal{I}\_{k}$ for the sake of example. In general we ask for $\mathcal{S}\_{k}$ to extract finite amount of information from function, $\mathcal{I}\_{k}$ to approximately restore original function from this information, and for $\mathcal{I}\_{k} \mathcal{S}\_{k}$ to converge to identity map.
2. The operator $\mathcal{S}\_{k}$ is analogous to encoder and $\mathcal{I}\_{k}$ - to decoder. Unlike encoder and decoder $\mathcal{S}\_{k}, \mathcal{I}\_{k}$ are not learned.
3. In current literature, $\mathcal{S}\_{k}$ is always a sampling operator. Given that, discretisation invariance architectures are mainly architectures agnostic to the resolution of the input.
4. We select $\mathcal{C}\_{[0, 1]}$ space with $L\_2$ norm for the sake of example. Function space and norm should be tailored to an intended application.
5. When $\mathcal{S}\_{k}$ and $\mathcal{I}\_{k}$ are selected we can have a family of maps $F\_{k}$ that always operate with finite amount of information for each $k$. Discretisation invariance is a requirement for the map $\mathcal{I}\_{k}F_{k}\mathcal{R}\_{k}$ to converge to a continuous operator $\mathcal{F}: \mathcal{C}\_{[0, 1]} \rightarrow \mathcal{C}\_{[0, 1]}$ between function spaces.

To show that discretisation invariant operators exist, we provide a simple example from numerical analysis. Integral $G(x_i) = \int_{0}^{x_i} g(x)dx$ can be approximated with Riemann sum $G(x_i) \simeq \sum_{j=1}^{i}g(x_j)/(N-1)$. We can represent this approximation with sampling operator and linear operators $F\_{k} given by $k\times k$ lower triangular matrices:

$$
\left(F_{k}\right)_{ij} = \left\{
    \begin{array}{ll}
        \frac{1}{k-1}, &  \text{if }i\leq j;\\
        0, & \text{ otherwise}.
    \end{array}
\right.
$$

For continuous function $g$, composition of sampling, Riemann sum and interpolation $\mathcal{I}\_{k}\left(F_{k}\left(\mathcal{S}\_{k}g\right)\right)$ converges to antiderivative $G(x) = \int\_{0}^{x}g(y)dy$.

## Examples and non-examples
An example with antiderivative operator suggests a general strategy to design discretisation invariant architectures: formulate all operations on functions in continuous form and use discretisation techniques from numerical analysis to process functions consistently on grids with different resolutions <d-cite key="berner2025principled"></d-cite>, <d-cite key="li2020fourier"></d-cite>, <d-cite key="li2020neural"></d-cite>. Most discretisation invariant architectures that we describe in this section follow this general recipe.

### Fourier Neural Operator
Fourier Neural Operator (FNO) is a most famous and successful example of discretisation invariant architecture <d-cite key="li2020fourier"></d-cite>. FNO is a feedforward neural network that uses three operations:
1. Convolution with kernel size $1$. For input functions $v^{i}(x)$ with $N$ "channels", the output is $\sum_{j=1}^{N}A_{ij}v^{j}(x)$.
2. Pointwise nonlinear activation.
3. Spectral convolution <d-cite key="rippel2015spectral"></d-cite> with truncation.

Spectral convolution is the only operation with spatial transfer of information. It can be understood either as parametrization of convolution in the Fourier domain where convolution operator becomes diagonal <d-cite key="rippel2015spectral"></d-cite>, or as an efficient evaluation of integral operator $\int \sum_{j} s_{ij}(y - x;\theta) v^{j}(y) dy$ with particular kernel convenient for implementation. The kernel is chosen to be periodic finite bandwidth function, so the whole integral operator can be implemented in three stages:
1. Fourier transform of the input with truncation $\hat{v}^{j}\_{k} = \mathcal{F}(v^{j}(x))\_{k},\,k=1,\dots,k\_{\max}$.
2. Linear operator diagonal in Fourier space $\hat{w}^{i}\_{m} = \sum\_{i} R\_{ijm} \hat{v}^{j}\_{m}$. Coefficients of tensor $R$ are learnable parameters.
3. Inverse Fourier transformation with padding to restore original spatial shape $\mathcal{F}^{-1}\left(\hat{w}^{i}_{m}\right)$.

Spectral convolution with truncation is discretisation invariant by construction, since it approximates continuous integral kernels with standard techniques from numerical analysis.

Many other architectures follow similar design pattern, e.g., <d-cite key="tripura2022wavelet"></d-cite>, <d-cite key="gupta2021multiwavelet"></d-cite>, <d-cite key="tran2021factorized"></d-cite>, by either modifying the parametrisation of spectral convolution or replacing Fourier with other fast transformations.

### Convolutional neural networks
Architecture based on convolutional neural networks (CNNs), especially ResNet <d-cite key="he2016deep"></d-cite> and U-Net <d-cite key="ronneberger2015u"></d-cite>, are highly successful for operator learning problems <d-cite key="stachenfeld2021learned"></d-cite>, <d-cite key="raonic2023convolutional"></d-cite>. They are often applied in a form of "image-to-image" mappings, with both images being physical fields of interest computed on the uniform grids.

{% include figure.liquid path="assets/img/2026-04-27-discretisation-invariance/receptive_field_conv.png" class="img-fluid" %}
<div class="caption">
    For a convolution operator with kernel size $5\times5$, a regular point collects information from $5$ neighbours along each dimension (shaded area). When the grid is refined the receptive field of convolution shrinks, so each point receives data from a smaller patch of $(x, y)$ space.
</div>

What makes convolutional architectures interesting in our context is their ability to process inputs of different resolutions. However, as illustrated in the image above, the receptive field of CNNs in coordinate space will decrease with the increase of resolution. We will see below, that when CNN is trained on fixed resolution, data on refined grid appears as out-of-distribution, leading to sharp drop in accuracy. As a result discretisation invariance is not observed.

### Deep Operator Network with linear observations
Deep Operator Network (DeepONet) is a meta-architecture <d-cite key="lu2019deeponet"></d-cite> based on the universal approximation results for operator learning <d-cite key="chen1995universal"></d-cite>. The architecture consists of two arbitrary neural networks: branch network and trunk network. For the input function $v(x)$, the output $u(x)$ is computed as follows:
1. Branch net $b$ takes whatever information about $v(x)$ is available (e.g., finite number of samples at selected points $v(x_1),\dots, v(x_d)$) and outputs a set of coefficients $c_{1}, \dots, c_{b}$.
2. The final layer of trunk net $t_{1}(x),\dots,t_{b}(x)$ provides a global basis that does not depend on the input $v(x)$.
3. The output of the architecture is constructed from branch and trunk nets $u(x) = \sum_{i=1}^{b} c_{i}t_{i}(x)$.

Readers familiar with reduced order modelling may recognise that the scheme closely resembles a non-intrusive proper orthogonal decomposition.

To make discretisation invariant DeepONet, we select a set of predefined basis function $\psi_1(x),\dots,\psi_m(x)$, and use them to form linear observations $o_i = \int \psi_{i}(x) v(x) dx$ which are later supplied to branch net. To compute linear observations, any numerical integration can be applied, e.g., [trapezoidal rule](https://en.wikipedia.org/wiki/Trapezoidal_rule).

### PCA-Net
PCA-Net is encoder-processor-decoder architecture based on proper-orthogonal decomposition (POD) or Karhunen–Loève expansion <d-cite key="hesthaven2018non"></d-cite>, <d-cite key="bhattacharya2021model"></d-cite>. For input $v(x)$ we compute output $u(x)$ as follows:
1. Encoder finds coefficients $c = \inf_{c} \left\|v(x) - \sum_{i=1}^{d} c_i\phi_i(x)\right\|_2^2$, where $\phi_i(x)$ are precomputed as explained below.
2. Processor is a standard feedforward architecture that transforms a vector of coefficients to another vector $d_i,i=1,\dots m$.
3. Similar to DeepONet, decoder computes a linear combination $u(x) = \sum_{i=1}^{m} d_i \psi_i(x)$, where $\psi_i(x)$ are computed similarly to $\phi_i(x)$.

For PCA-Net functions $\phi\_i(x)$, $\psi\_i(x)$ are computed using POD <d-cite key="volkwein2013proper"></d-cite>. Let $v_{j}(x),\,j=1,\dots,N_{\text{train}}$ be inputs from the train set. Basis functions are recursively defined as solutions to optimisation problems

$$
\psi_i(x) = \arg \min_{\psi} \sum_{j}\left\|v_{j} - \psi\left(\psi, v_{j}\right)\right\|_{2}^{2},\text{ subject to }\left\|\psi\right\|_2 = 1,\left(\psi_{k}, \psi\right) = 0,\text{ for }k < i.
$$

That is, precisely the same way as principal components in PCA, but in functional space.

### Neural fields
This is another example of encoder-processor-decoder architecture where both input and output are approximated by a form of implicit neural representation <d-cite key="serrano2023operator"></d-cite>. Operators based on neural fields work precisely as PCA-Net but uses different approach to represent functions by finite-dimensional vectors.

To illustrate how basis functions are built, suppose we collected a dataset of inputs $v_i(x),\,i=1,\dots,N_{\text{train}}$. We select a neural network with weights $\theta$, that take coordinate $x$ in the first layer, and, in addition, vector $z$ in some hidden layer. We find parameters of the resulting architecture $\phi_{\theta}(x;z)$ by optimising the loss

$$
\min_{\theta} \left(\sum_{i=1}^{N} \min_{z_i}\left\|\phi_{\theta}(x;z_i) - v_{i}(x)\right\|_{2}^2\right).
$$

As a result for all dataset we will compute global parameters $\theta$ that are shared among samples $v_{i}$, and for each individual sample we find a coding vector $z_{i}$. This finite coding vector is used as a representation of function $v_i(x)$. For new inputs outside of the training set, the optimization problem above is solved with fixed $\theta$ to find finite-dimensional representation $z$. The same is done for the targets, and after that we are left with the problem of learning maps between finite dimension spaces.

Note, that all operations in this scheme are formulated with no explicit discretisation, ensuring that the whole architecture is discretisation agnostic.

### Graph neural networks
Unstructured grids are very common in scientific computing, especially when complex geometries are involved. Given that, graph neural networks (GNNs) are a natural choice for building neural PDE solvers <d-cite key="brandstetter2022message"></d-cite>. GNN is an example of architecture that can handle variations in grid and geometry, but nonetheless is not discretisation invariant.

{% include figure.liquid path="assets/img/2026-04-27-discretisation-invariance/receptive_field_GNN.png" class="img-fluid" %}
<div class="caption">
    An example of the receptive field for GNN. Similarly to CNN, receptive field shrinks when resolution is increased.
</div>

The reason GNNs are not discretisation invariant is precisely the same as for CNNs: the architecture can be applied on a refined grid, but the receptive field is going to shrink leading to out of distribution inputs. The change in receptive field is illustrated in the picture above.

### Graph kernel networks
Graph kernel network (GKN) is a discretisation agnostic version of GNN <d-cite key="li2020neural"></d-cite>. It replaces message passing by integral operator
$$
v_{i+1}(x, y) = \int_{B(x, y)} k_{\phi}(x, y, x^{'}, y^{'}, u(x, y), u(x^{'}, y^{'})) v_{i}(x^{'}, y^{'}) dx^{'} dy^{'},
$$

where $v_{i+1}(x, y)$ is the output of the layer, $v_{i}(x^{'}, y^{'})$ is an input, $B(x, y)$ is a ball of predefined radius around $x, y$, and $u(x, y)$ is an input to the network, e.g., a diffusivity coefficient in stationary diffusion equation.

{% include figure.liquid path="assets/img/2026-04-27-discretisation-invariance/receptive_field_KNN.png" class="img-fluid" %}
<div class="caption">
    For graph kernel networks, the receptive field, defined in $(x, y)$ space, is a hyperparameter of the architecture. When the grid is refined a graph of nearest neighbours is recomputed, ensuring discretisation invariance of the architecture.
</div>

A convenient way to approximate integral above is to use Monte Carlo method
$$
v_{i+1}(x) = \frac{1}{N_{mc}}\sum_{l=1}^{N_{mc}} k_{\phi}(x, y, x^{'}_{l}, y^{'}_{l}, u(x, y), u(x^{'}_{l}, y^{'}_{l})) v_{i}(x^{'}_{l}, y^{'}_{l}),
$$

where $(x\_{l}^{'}, y\_{l}^{'})$ are points inside a ball $B(x, y)$ as shown in the figure above.

Importantly, the radius of the ball $B(x, y)$ is not related to discretisation used, and because of that when the grid is refined the finite sum approximates the same integral using more terms.

## Training and evaluation
We demonstrate discretisation invariance on three architectures and two PDEs. The code is available [in this repository](https://github.com/4gnskq5g2s-collab/discretisation_invariance). Software used include jax <d-cite key="deepmind2020jax"></d-cite> and equinox <d-cite key="kidger2021equinox"></d-cite>.

The first PDE is Burgers equation

$$
\frac{\partial u(x, t)}{\partial t} + \frac{1}{2}\frac{\partial \left(u(x, t)\right)^2}{\partial x} = \nu \frac{\partial^2 u(x, t)}{\partial x^2},\,u(0, t) = u(1, t) = 0,
$$

with initial conditions sampled from Gaussian random field and viscosity $\nu = 0.1$.

For Burgers equation neural networks were trained to predict $u(x, 0.3)$ from initial condition $u(x, 0)$. The examples of features and targets are on the figure below.

{% include figure.liquid path="assets/img/2026-04-27-discretisation-invariance/Burgers_samples.png" class="img-fluid" %}
<div class="caption">
    Samples from Burgers dataset: initial conditions $u(x, 0)$ in the first row, solution $u(x, 0.3)$ in the second row.
</div>

The second dataset is based on stationary diffusion equation

$$
-\frac{d}{dx}\left(k(x) \frac{d \phi(x)}{dx}\right) = 0,\,\phi(0) = \phi(1) = 0,
$$

with $k(x)$ sampled from Gaussian random field with transformation that ensures: (i) $k(x) > 0$, (ii) large spatial variability.

The task for this equation was to predict $u(x)$ from $k(x)$. Samples from the dataset are below.

{% include figure.liquid path="assets/img/2026-04-27-discretisation-invariance/diffusion_samples.png" class="img-fluid" %}
<div class="caption">
    Samples from diffusion dataset: diffusivity coefficients $k(x)$ in the first row, solutions $\phi(x)$ in the second row.
</div>

On these two datasets we train three architectures: FNO, U-Net and DeepONet with linear observations. The setup of experiment is a standard one used to demonstrate discretisation invariance <d-cite key="li2020fourier"></d-cite>:
1. We generate dataset on grid with $N=512$ points.
2. Neural network is trained on downsampled version with $N=64$ points.
3. After training it is evaluated on grids with $64, 128, 256, 512$ points.

The results of the experiments are reported below.

{% include figure.liquid path="assets/img/2026-04-27-discretisation-invariance/discretisation_invariance.png" class="img-fluid" %}
<div class="caption">
    Relative test error for neural networks trained on grids with $64$ points and evaluated on grids with higher resolutions. Discretisation invariant architectures, DeepONet and FNO, show mild variations of relative error when resolution increases. In contrast, relative error of U-Net instantly reaches $>100\%$ when resolution increases.
</div>

It is quite clear that discretisation invariant architecture tolerate resolution increase much better than U-Net. Accuracy of DeepONet even slightly improves with resolution. The accuracy of FNO slowly decreases.

## The role of discretisation invariance
We have seen the definition of discretisation invariance and typical numerical demonstration that shows the difference between classical and discretisation agnostic architectures. It is not hard to find dozens of papers that develop novel discretisation invariant architectures and confirm it by experiments alike one we reproduce above. The accuracy is still important, but discretisation invariance becomes a goal in itself. Clearly, for some reason we want to develop discretisation invariant architectures. Unfortunately, it is rarely explained *why* we need this. We would like to invite research community to a discussion on the end of discretisation invariance. We provide arguments against discretisation invariance and hope that other researchers will publish arguments for discretisation invariance elsewhere.

**Discretisation invariance is an asymptotic property.**

From the definition, and from typical evaluation strategy, discretisation invariance is a statement about convergence <d-cite key="azizzadenesheli2024neural"></d-cite>. Convergence is important in scientific computing, because it allows one to reach arbitrary accuracy on paper and very high precision on digital computer <d-cite key="bailey2012high"></d-cite>. In contrast, for operator learning problem error rarely drops below $10^{-4}$, and given that train data has finite resolution it is unlikely that accuracy increases beyond what was reached on the training stage.

**Functional data analysis. All data is finite-dimensional.**

Functional data analysis (FDA) is there for quite some time <d-cite key="ramsay1991some"></d-cite>. If we look at the literature on FDA, we find that the objects of interest are functions, and researchers in FDA study classification, regression and interpolation in Banach spaces. Given that, FDA is very related to operator learning and discretisation invariance, albeit researchers in these two fields rarely reference each other. Peculiarly enough, FDA is not popular in the modern applications of machine learning. Some reasons for that are summarised in [the illuminating discussion](https://stats.stackexchange.com/a/564607) on StackExchange. One argument is all the observations we have are finite, and it is often possible to come up with an algorithm for finite data that outperforms FDA algorithms.

**Discretisation invariance is not a good indicator of performance.**

In <d-cite key="berner2025principled"></d-cite> authors argue that essentially any architecture can be made discretisation invariant. An interesting consequence of this claim is discretisation invariance ceases to be a good guiding principle for architecture design. Some architecture perform well, some perform poorly, but discretisation invariance has nothing to do with that, since any architecture is discretisation invariant after mild adjustments. Lets consider FNO as an example.

FNO is certainly a great architecture, but why? Is it because of discretisation invariance? The main component of FNO is spectral convolution, i.e., a convolution layer parametrised in the Fourier space. In the paper <d-cite key="rippel2015spectral"></d-cite>, where it was introduced, authors observed that for image classification problem (problem with fixed resolution) architectures with spectral convolutions and pooling converge $2$ to $5$ times faster and lead to improved accuracy in comparison with classical CNNs. Why is that? Whatever the answer is, it unlikely invokes discretisation invariance.

**When trained on grid with fixed resolution, classical architectures often perform better.**

As we already discussed in the section on FDA, methods specialised for finite-dimensional data are often perform better. An apt example is deterministic weather forecast where transformer-based models are at the top of the leaderboard <d-cite key="liu2024evaluation"></d-cite>. To be fair, discretisation invariant architectures are also comparable or superior to classical weather prediction models  <d-cite key="pathak2022fourcastnet"></d-cite>, but whether discretisation invariance has a role in its success is unclear.

**Use cases for discretisation invariance are slim.**

A popular neural operator library [list many advantages of the approach](https://neuraloperator.github.io/dev/theory_guide/advantages.html). Some of them are clearly misleading, e.g., "The approximation quality improves as the input resolution increases, with the error vanishing in the limit of infinite resolution." This is not what is observed, in practice approximation quality deteriorates with resolution or, at best, saturates. Some are overly optimistic "Once trained, neural operators can produce high-resolution solutions much faster than traditional numerical methods, often achieving speedups of 100-1,000,000x!" See <d-cite key="mcgreivy2024weak"></d-cite> for the impartial evaluation of neural PDE solvers. Importantly, authors of the library do not provide any references supporting their claims.

A central motif of many stated advantages is the benefit of processing data with different resolutions consistently. In particular, authors suggest that one can train neural operator on low resolution data and finetune on high resolution data. We believe this statement is accurate and it should be possible to speed up training that way. This is the only use case of discretisation invariance we can come up with.

---
