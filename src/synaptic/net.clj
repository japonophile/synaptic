(ns
  ^{:doc "synaptic - neural network architecture"
    :author "Antoine Choppin"}
  synaptic.net
  (:require [clatrix.core :as m]
            [synaptic.datasets :as d]
            [synaptic.util :as u])
  (:import  [synaptic.datasets DataSet]
            [java.io File FileWriter]
            [clatrix.core Matrix])
  (:gen-class))


(defrecord Arch [layers])
(defrecord Net [arch weights training])

; Network architecture

(defmulti layer-units
  "Returns the number of units (outputs) in a given layer of the neural network,
  based on the net architecture (note: layer 0 is the input layer)."
  (fn [layers l] (:type (nth layers l))))

(defmethod layer-units
  :input
  [layers l]
  {:pre [(== 0 l)]}
  (let [layer (nth layers l)]
    (or (:n layer)
        (apply * (:fieldsize layer)))))

(defmulti layer-instrument
  "Instrument a layer"
  (fn [layers l] (:type (nth layers l))))

(defmethod layer-instrument
  :default
  [layers l]
  (nth layers l))

(defmethod layer-instrument
  :input
  [layers l]
  (let [layer (nth layers l)]
    (if-let [fieldsize (:fieldsize layer)]
      (let [fieldsize3d (if (= 2 (count fieldsize))
                          (into [1] fieldsize)
                          fieldsize)]
        (assoc layer :fieldsize fieldsize3d))
      layer)))

(defn instrument-layers
  "Instrument neural network layers."
  [layers]
  {:pre [(< 1 (count layers))]}
  ; implementation note: we are not just doing a map operation on layers
  ; because instrumentation of layer l+1 relies on instrumentation of layer l
  (loop [l 0, layers layers]
    (if (< l (count layers))
      (recur (inc l) (assoc layers l (layer-instrument layers l)))
      layers)))

(defn layer-actfn-kind
  "Returns the activation function to be used by a given layer of the neural network,
  based on the net architecture."
  [layers l]
  {:pre [(< 0 l (count layers))]}
  (:act-fn (nth layers l)))

(defn layer-actfn
  "Returns the activation function for a network at a given layer (or last layer)."
  [layers l]
  (let [kind (layer-actfn-kind layers l)]
    (resolve (symbol "synaptic.net" (name kind)))))

; Weight initialization

(defmulti init-weights
  "Initialize weights of a single network layer."
  (fn [layers l] (:type (nth layers l))))

(defn init-all-weights
  "Initialize weights of all layers."
  [layers]
  {:pre [(< 1 (count layers))]}
  (vec (for [i (range (dec (count layers)))]
         (init-weights layers (inc i)))))

(defn weight-histograms
  "Compute histograms of each layer's weights (excluding bias)."
  [weights & [nbins]]
  (vec (pmap #(u/histogram (flatten (u/without-bias %)) (or nbins 20)) weights)))

; Activation functions

(defn linear
  "Linear activation function.
  Returns the logit."
  [z]
  z)

(defn binary-threshold
  "Binary threshold activation function.
  Returns 1 if the logit is >= 0, or 0 otherwise."
  [zs]
  (m/map (fn [z] (if (>= z 0) 1 0)) zs))

(defn bipolar-threshold
  "Binary threshold activation function.
  Returns 1 if the logit is >= 0, or -1 otherwise."
  [zs]
  (m/map (fn [z] (if (>= z 0) 1 -1)) zs))

(defn sigmoid
  "Sigmoid activation function.
  Computed as 1 / (1 + e^(-z))."
  [zs]
  (m/div 1.0 (m/+ 1.0 (m/exp (m/- zs)))))

(defn hyperbolic-tangent
  "Hyperbolic tangent activation function."
  [zs]
  (m/tanh zs))

(defn sample-with-p
  "Returns a random sample with probability p."
  [ps]
  (m/map (fn [p] (if (< (u/random) p) 1 0)) ps))

(defn binary-stochastic
  "Binary stochastic activation function.
  Returns a random sample with probability given by the logistic (sigmoid)
  of the logit."
  [zs]
  (sample-with-p (sigmoid zs)))

(defn binary-prob
  "Binary probability activation function.
  Returns 1 if the sigmoid of the logit is >= 0.5, or 0 otherwise."
  [zs]
  (m/map (fn [z] (if (< z 0.5) 0 1)) (sigmoid zs)))

(defn softmax
  "Softmax activation function.
  Returns e(z) / Sum[e(z)]."
  [zs]
  (let [k    (m/ncols zs)
        nzs  (m/- zs (m/* (m/matrix (mapv (partial apply max) (m/rows zs)))
                          (m/ones 1 k)))
        es   (m/exp nzs)
        sums (m/* (m/matrix (mapv (partial reduce +) es))
                  (m/ones 1 k))]
    (m/div es sums)))

(defn actfn-kind
  "Returns the kind of activation function (expressed as a keyword) for a network 
  at a given layer (or last layer)."
  [^Net nn & [l]]
  (let [layers (-> nn :arch :layers)]
    (layer-actfn-kind layers (or l (dec (count layers))))))

(defn act-fn
  "Returns the activation function for a network at a given layer (or last layer)."
  [^Net nn & [l]]
  (let [kind (actfn-kind nn l)]
    (resolve (symbol "synaptic.net" (name kind)))))

; Constructor

(defn neural-net
  "Create a neural network, and returns an atom pointing to it.

  Takes as input a seq with description of each layer and (optionally)
  an initialized training.  See (doc training) for details.

  The description of each layer is a map with the following parameters:
    :type        :input, :fully-connected or :convolution
    :n           the number of units in the layer (required for fully-connected
                   layers, and for the input layer if no :fieldsize is specified)
    :fieldsize   (only for the input layer) the field size [w h], i.e. the width
                   and height of an input sample.  it can also be 3-dimensional,
                   i.e. [k w h] where k is the number of 'channels' per sample
    :act-fn      the activation function; e.g. :sigmoid, :softmax etc.
                   (required for hidden & output layers)
    :feature-map (only for convolution layers) description of the feature map
                   including :size and :k (number of feature maps)
    :pool        (only for convolution layers) a map defining the :kind of pooling
                   (:avg or :max) and the pool :size
  
  Examples:
  * (neural-net [{:type :input :n 100}
                 {:type :fully-connected :n 20 :act-fn :sigmoid}
                 {:type :fully-connected :n 10 :act-fn :softmax}])
    a network with 100 input units, 20 hidden (fully-connected) units using
    the sigmoid activation function and 10 output (fully-connected) units using
    the softmax activation function.
    note: the same network can be created by (mlp [100 20 10] [:sigmoid :softmax])
  
  * (neural-net [{:type :input :fieldsize [3 28 28]}
                 {:type :convolution :feature-map {:size [9 9] :k 6}
                  :act-fn :hyperbolic-tangent}
                 {:type :fully-connected :n 10 :act-fn :softmax}])
    a network accepting 3-channel (e.g. R,G,B) 28*28 input images (hence with
    3 * 28 * 28 = 2352 inputs), followed by a convolution layer of 6 9x9 feature
    maps using the tanh activation function, followed by a fully-connected output
    layer with 10 softmax units. 

  * (neural-net [{:type :input :fieldsize [28 28]}
                 {:type :convolution :feature-map {:size [9 9] :k 6}
                  :act-fn :hyperbolic-tangent :pool {:kind :max :size [2 2]}}
                 {:type :fully-connected :n 10 :act-fn :softmax}])
    the same network as above, but with single-channel input samples (e.g. grayscale
    images) and with max pooling of size 2x2 in the convolution layer."
  [layers & [training]]
  {:pre [(vector? layers)]}
  (let [layers (instrument-layers layers)]
    (atom (Net. (Arch. layers) (init-all-weights layers) training))))

(defn save-neural-net
  "Save neural network to disk.
  The header is saved separately for quicker access."
  [^Net nn]
  (if-let [nnname (-> nn :header :name)]
    (u/save-data "neuralnet" nn nnname)
    (u/save-data "neuralnet" nn)))

(defn load-neural-net
  "Load a neural net from disk."
  [nnname]
  (u/load-data "neuralnet" nnname))

; Network outputs (neuron activities, forward pass)

(defmulti layer-activities
  "Compute activities of a layer of neurons, with the specified activation
  function actfn, given the layer weights w and layer inputs x."
  (fn [layers l w x] (:type (nth layers l))))

(defn valid-inputs?
  "Returns true if x are valid inputs for the neural network, else otherwise."
  [^Net nn x]
  {:pre [(= Net (type nn))]}
  (and (m/matrix? x)
       (= (second (m/size x))
          (layer-units (-> nn :arch :layers) 0))))

(defn net-activities
  "Compute activities of all layers of a neural network, for the given inputs x."
  [^Net nn x]
  {:pre [(= Net (type nn)) (valid-inputs? nn x)]}
  (let [layers (-> nn :arch :layers)]
    (loop [l 1, x x, ws (:weights nn), as []]
      (if (first ws)
        (let [a (layer-activities layers l (first ws) x)]
          (recur (inc l) (:a a) (rest ws) (conj as a)))
        as))))

(defn estimate
  "Estimate classes for a given data set, by computing network output for each
  sample of the data set, and returns the most probable class (label) - or its
  index if labels are not defined."
  [^Net nn ^DataSet dset]
  (let [x  (:x dset)
        y  (m/dense (:a (last (net-activities nn x))))
        n  (count (first y))
        ci (mapv #(apply max-key % (range n)) y)
        cs (-> nn :header :labels)]
    (if cs
      (mapv #(get cs %) ci)
      ci)))

