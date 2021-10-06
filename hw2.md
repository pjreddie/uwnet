# CSE 490g1 / 599g1 Homework 2 #

Welcome friends,

For the third assignment we'll be implementing a powerful tool for improving optimization, batch normalization!

You'll have to copy over your answers from the previous assignment.

## 7. Batch Normalization ##

The idea behind [batch normalization](https://arxiv.org/pdf/1502.03167.pdf) is simple: we'll normalize the layer output so every neuron has zero mean and unit variance. However, this simple technique provides huge benefits for model stability, convergence, and regularization.

### Batch Norm and Convolutions ###

Batch normalization after fully connected layers is easy. You simply calculate the batch statistics for each neuron and then normalize. With our framework, every row is a different example in a batch and every column is a different neuron so we will calculate statistics for each column and then normalize so that every column has mean 0 and variance 1.

With convolutional layers we are going to normalize the output of a filter over a batch of images. Each filter produces a single channel in the output of a convolutional layer. Thus for batch norm, we are normalizing across a batch of channels in the output. So, for example, we calculate the mean and variance for all the 1st channels across all the examples, all the 2nd channels across all the examples, etc. Another way of thinking about it is we are normalizing the output of a single filter, which gets applied both to all of the examples in a batch but also at numerous spatial locations for every image.

Thus for our batch normalization functions we will be normalizing across rows but also across columns depending on the spatial component of a feature map. Check out `batch_norm.c`, I've already filled in the `mean` example for calculating the mean of a batch.

The `groups` parameter will tell you how many groups (i.e. channels) there are in the output. So, if your convolutional layer outputs a `32 x 32 x 8` image and has a batch size of 128, the matrix `x` will have 128 rows and 8192 columns. We want to calculate a mean for every channel thus the `groups` parameter will be 8 and our matrix `m` will have 1 row and 8 columns (since there are 8 channels in the output).

We also calculate an `n` parameter that tells us the number of elements per group in one example. The images we are processing are `32 x 32` so the `n` parameter in this case will be the integer 1024. The total number of elements in matrix `x` is the number of examples in the batch (`x.rows`) times the number of groups (`groups`) times the number of elements per group (`n`).

After a fully connected layer, the `groups` parameter would always be the number of outputs and we would calculate separate means for each neuron in the output and the `n` parameter would be 1.

### Forward propagation ###

These are the forward propagation equations from the [original paper](https://arxiv.org/abs/1502.03167). Note, in the original terminology we're just use `x̂` as the output, we'll skip the scaling and shifting:

<a href="https://www.codecogs.com/eqnedit.php?latex=\dpi{200}&space;\large&space;\begin{align*}&space;\mu&space;&=&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;x_i&space;\\&space;\sigma^2&space;&=&space;\frac{1}{n}\sum_{i=1}^{n}(x_i&space;-&space;\mu)^2&space;\\&space;y_i&space;&=&space;\frac{x_i&space;-&space;\mu}{\sqrt{\sigma^2&space;&plus;&space;\epsilon}}&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\dpi{200}&space;\large&space;\begin{align*}&space;\mu&space;&=&space;\frac{1}{n}&space;\sum_{i=1}^{n}&space;x_i&space;\\&space;\sigma^2&space;&=&space;\frac{1}{n}\sum_{i=1}^{n}(x_i&space;-&space;\mu)^2&space;\\&space;y_i&space;&=&space;\frac{x_i&space;-&space;\mu}{\sqrt{\sigma^2&space;&plus;&space;\epsilon}}&space;\end{align*}" title="\large \begin{align*} \mu &= \frac{1}{n} \sum_{i=1}^{n} x_i \\ \sigma^2 &= \frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2 \\ y_i &= \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}} \end{align*}" /></a>

### 7.1 `variance` ###

Fill in the section to compute the variance of a feature map. As in the `mean` computation, we will compute variance for each filter. We need the previously computed `mean` for this computation so it is passed in as a parameter. Remember, variance is just the average squared difference of an element from the mean:

![variance equation: Var(X) = 1/n \sum_{x=1}^{n} (x_i - \mu)^2](https://wikimedia.org/api/rest_v1/media/math/render/svg/0c5c6e7bbd52e69c29e2d5cfe21989313aba55d4)

Don't take the square root just yet, that would be standard deviation!

### 7.2 `normalize` ###

To normalize our output, we simply subtract the mean from every element and divide by the standard deviation (now you'll need a square root). When you're dividing by the standard deviation it can be wise to add in a small term (the epsilon in the batchnorm equations) to prevent dividing by zero. Especially if you are using RELUs, you may occassionally have a batch with 0 variance.

You should use `eps = 0.00001f`.

### Understanding the forward pass ###

`batch_normalize_forward` shows how we process the forward pass of batch normalization. Mostly we're doing what you'd expect, calculating mean and variance and normalizing with them.

We are also keeping track of a rolling average of our mean and variance. During training or when processing large batches we can calculate means and variances but if we just want to process a single image it can be hard to get good statistics, we have to batch to norm against! Thus if we have a batch size of 1 we normalize using our rolling averages.

We assume the `l.rolling_mean` and `l.rolling_variance` matrices are initialized when the layer is created.

We also have the matrix pointer `l.x` which will keep track of the input to the batch norm process. We will need to remember this for the backward step!

### Backward propagation ###

The backward propagation step looks like this:

<a href="https://www.codecogs.com/eqnedit.php?latex=\large&space;\begin{align*}&space;\frac{dL}{d\mu}&space;&=&space;\sum_{i=1}^{n}&space;\frac{dL}{dy_i}\cdot\frac{-1}{\sqrt{\sigma^2&space;&plus;&space;\epsilon}}&space;\\&space;\frac{dL}{d\sigma^2}&space;&=&space;\sum_{i=1}^{n}&space;\frac{dL}{dy_i}\cdot(x_i&space;-&space;\mu)\cdot&space;\frac{-1}{2}(\sigma^2&space;&plus;&space;\epsilon)^{-3/2}&space;\\&space;\frac{dL}{dx_i}&space;&=&space;\frac{dL}{dy_i}&space;\cdot&space;\frac{1}{\sqrt{\sigma^2&space;&plus;&space;\epsilon}}&space;&plus;&space;\frac{dL}{d\sigma^2}&space;\cdot&space;\frac{2(x_i&space;-&space;\mu)}{n}&space;&plus;&space;\frac{dL}{d\mu}\cdot&space;\frac{1}{n}&space;\end{align*}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\large&space;\begin{align*}&space;\frac{dL}{d\mu}&space;&=&space;\sum_{i=1}^{n}&space;\frac{dL}{dy_i}\cdot\frac{-1}{\sqrt{\sigma^2&space;&plus;&space;\epsilon}}&space;\\&space;\frac{dL}{d\sigma^2}&space;&=&space;\sum_{i=1}^{n}&space;\frac{dL}{dy_i}\cdot(x_i&space;-&space;\mu)\cdot&space;\frac{-1}{2}(\sigma^2&space;&plus;&space;\epsilon)^{-3/2}&space;\\&space;\frac{dL}{dx_i}&space;&=&space;\frac{dL}{dy_i}&space;\cdot&space;\frac{1}{\sqrt{\sigma^2&space;&plus;&space;\epsilon}}&space;&plus;&space;\frac{dL}{d\sigma^2}&space;\cdot&space;\frac{2(x_i&space;-&space;\mu)}{n}&space;&plus;&space;\frac{dL}{d\mu}\cdot&space;\frac{1}{n}&space;\end{align*}" title="\large \begin{align*} \frac{dL}{d\mu} &= \sum_{i=1}^{n} \frac{dL}{dy_i}\cdot\frac{-1}{\sqrt{\sigma^2 + \epsilon}} \\ \frac{dL}{d\sigma^2} &= \sum_{i=1}^{n} \frac{dL}{dy_i}\cdot(x_i - \mu)\cdot \frac{-1}{2}(\sigma^2 + \epsilon)^{-3/2} \\ \frac{dL}{dx_i} &= \frac{dL}{dy_i} \cdot \frac{1}{\sqrt{\sigma^2 + \epsilon}} + \frac{dL}{d\sigma^2} \cdot \frac{2(x_i - \mu)}{n} + \frac{dL}{d\mu}\cdot \frac{1}{n} \end{align*}" /></a>

So to backward propagate we'll need to calculate these intermediate results, dL/dmu and dL/dsigma^2. Then, using them, we can calculate dL/dx.

### 7.3 `delta_mean` ###

Calculate dL/dmu.

### 7.4 `delta_variance` ###

Calculate dL/dsigma^2.

### 7.5 `delta_batch_norm` ###

Using the intermediate results, calculate dL/dx.

### 7.6 Using your batchnorm ###

Try out batchnorm! To add it after a layer, just make this simple change:

    make_convolutional_layer(16, 16, 8, 16, 3, 1)
    make_batchnorm_layer(16) # groups parameter should be same as output channels from previous layer
    make_activation_layer(RELU)

You should be able to add it after convolutional or connected layers. The standard for batch norm is to use it at every layer except the output. First, train the `conv_net` as usual. Then try it with batchnorm. Does it do better??

In class we learned about annealing your learning rate to get better convergence. We ALSO learned that with batch normalization you can use larger learning rates because it's more stable. Increase the starting learning rate to `.1` and train for multiple rounds with successively smaller learning rates. Using just this model, what's the best performance you can get?

## PyTorch Section ##

Upload `hw2.ipynb` to Colab and train a neural language model.

## Turn it in ##

First run the `collate_hw2.sh` script by running:

    bash collate_hw2.sh
    
This will create the file `hw2.tar.gz` in your directory with all the code you need to submit. The command will check to see that your files have changed relative to the version stored in the `git` repository. If it hasn't changed, figure out why, maybe you need to download your ipynb from google?

Submit `hw2.tar.gz` in the file upload field for Homework 2 on Canvas.

