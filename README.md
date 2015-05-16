# synaptic

[Synaptic](http://chopp.in/clj/synaptic/) is a Neural Networks library written in Clojure.

It is intendend to be used for experimenting with various neural network
architectures and learning algorithms, as well as to solve real-life
problems.

It is largely inspired by Geoffrey Hinton's class "Neural Networks
for Machine Learning" available online on
[Coursera](https://class.coursera.org/neuralnets-2012-001).

## Features

Synaptic allows to create multilayer feedforward networks and train them
using various algorithms such as:
- perceptron learning rule
- backpropagation
- L-BFGS (approximation of Newton's method)
- R-prop
- RMSprop
- L1 & L2 regularization
- convolution / pooling layers

It also allows to play with various training parameters like learning
rate and momentum, and supports adaptive learning rate and variants of
the momentum method (Nesterov's momentum).

## Usage

To use Synaptic, first add this to your `project.clj`:

[![Clojars Project](http://clojars.org/synaptic/latest-version.svg)](http://clojars.org/synaptic)

You can then experiment in the REPL.

First, make sure you have a directory called `data` in your project, where you copy the 
training set (you can use the mnist10k training set provided in Synaptic git repo).

```
$ ls data/
data/trainingset.mnist10k
```

Next, open your REPL, load the training set and create a neural net, as follows:

```clojure
(require '[synaptic.core :refer :all])

(def trset (load-training-set "mnist10k"))   ; load MNIST training set
(def net (mlp [784 100 10]             ; net with 784 inputs, 100 hidden units
              [:sigmoid :softmax]      ; and 10 (softmax) outputs
              (training :backprop)))   ; to be trained with backpropagation

(train net trset 10)              ; train the network for 10 epochs (returns a future)
@*1                               ; (deref to wait for the future to complete)

(-> @net :training :stats)        ; show training statistics
```

Note: you may want to do `export JVM_OPTS="-Xmx1024m"` before starting your
REPL to make sure you have enough memory to load the training set.

Synaptic github repository includes a subset of the
[MNIST handwritten digit](http://yann.lecun.com/exdb/mnist/)
training data available on Yann LeCun's website, so you can experiment.

But you can also easily create your own training set:

```clojure
(def trset (training-set samples labels))
```

There are many options you can specify to customize the training algorithm to
your taste.

## Examples

```clojure
(require '[synaptic.core :refer :all])

(def trset (load-training-set "mnist10k"))
```

- Classic perceptron (has only 1 layer and uses misclassification cost function 
instead of cross-entropy).

```clojure
(def net (mlp [784 10] :binary-threshold (training :perceptron)))
```

- Multi-layer perceptron with 784 input units, 100 hidden units, and 10 output units.  Initialize backpropagation training with a learning rate of 0.001:

```clojure
(def net (mlp [784 100 10] [:sigmoid :softmax]
              (training :backprop {:learning-rate {:epsilon 0.001}})))
```

- Neural net with an input layer of 28x28 fields, a convolution layer with 3 9x9 fieldmaps and 2x2 pooling, and a fully connected layer with 10 output units.

```clojure
(def net (neural-net [{:type :input :fieldsize [28 28]}
                      {:type :convolution :act-fn :hyperbolic-tangent
                       :feature-map {:size [9 9] :k 3}
                       :pool {:kind :max :size [2 2]}}
                      {:type :fully-connected :act-fn :sigmoid :n 10}]
                     (training :backprop {:learning-rate {:epsilon 0.01}})))
```

- Specify adaptive learning rate with min and max gain of 0.01 and 10.0:

```clojure
(def net (mlp [784 100 10] [:sigmoid :softmax]
              (training :backprop
              {:learning-rate {:epsilon 0.01 :adaptive true
                               :ming 0.01 :maxg 10.0}})))
```

- Use L-BFGS unconstrained optimization algorithm instead of backprop:

```clojure
(def net (mlp [784 100 10] [:sigmoid :softmax] (training :lbfgs)))
```

## How to monitor training

As mentioned above, the `train` function returns a future which will be
completed when the training stops.  You can monitor the training by adding
a watch on the `net` (it is an atom) so you can see when its weights change
or when training state or training stats are updated (typically every epoch).

## To Do

- Implement other regularization techniques, such as weight decay

- Support other kind of neural networks, such as RNN or RBM

## Feedback

This library is still in its infancy.
Your feedback on how it can be improved, and contributions are welcome.

## License

Copyright © 2014-2015 Antoine Choppin

Distributed under the Eclipse Public License, the same as Clojure.

