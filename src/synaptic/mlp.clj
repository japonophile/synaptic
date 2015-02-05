(ns
  ^{:doc "synaptic - multi-layer perceptrons"
    :author "Antoine Choppin"}
  synaptic.mlp
  (:require [clatrix.core :as m]
            [synaptic.net :refer :all]
            [synaptic.training :refer :all]
            [synaptic.util :as u])
  (:gen-class))


; Network architecture

(defmethod layer-units
  :fully-connected
  [layers l]
  {:pre [(< 0 l (count layers))]}
  (:n (nth layers l)))


; Weight initialization

(defmethod init-weights
  :fully-connected
  [layers l]
  {:pre [(< 0 l (count layers))]}
  (let [nin  (layer-units layers (dec l))
        nout (layer-units layers l)]
    (m/matrix (for [i (range nout)]
                (for [j (range (inc nin))]
                  (* (/ 1.0 nin) (u/nrand)))))))


; Constructor

(defn mlp
  "Create a Multi-Layer Perceptron, that is a multi-layer neural net,
  with fully-connected layers.
  This is just a shortcut for the neural-net constructor.
  
  Takes as input a seq with the number of units in each layer,
  and the activation function of network units (as a symbol).
  
  Ex. (mlp [10 5 1] :sigmoid)
    creates a network with 10 input units, 5 hidden units and 1 output unit,
    using the sigmoid activation function.
  
  It is possible to define an activation function per layer.
  Ex. (mlp [10 20 10 5] [:sigmoid :sigmoid :softmax])
  (There is no activation function for the input layer).
  
  It is also possible to initialize the training for the network.
  Ex. (mlp [100 20 50] :sigmoid (training :backprop))
  See (doc training) for details."
  [nunits actfn & [training]]
  {:pre [(vector? nunits) (or (keyword? actfn) (= (dec (count nunits)) (count actfn)))]}
  (let [actfns (if (keyword? actfn) (repeat actfn) actfn)
        layers (into [{:type :input :n (first nunits)}]
                     (mapv (fn [n a] {:type :fully-connected :n n :act-fn a})
                           (rest nunits) actfns))]
    (neural-net layers training)))


; Network outputs

(defmethod layer-activities
  :fully-connected
  [layers l w x]
  {:pre [(m/matrix? w) (m/matrix? x)
         (= (second (m/size w)) (inc (second (m/size x))))]}
  (let [actfn (layer-actfn layers l)]
    {:a (actfn (m/* (u/with-bias x) (m/t w)))}))


; Backpropagation learning

(defmethod prev-layer-error-deriv-wrt-logit
  :fully-connected
  [layers l dEdz-out ay w]
  {:pre [(> l 1)]} ;l = layer after the 1st hidden layer (exclude input layer)
  (let [y    (:a ay)
        dEdy (m/* dEdz-out (u/without-bias w))
        dydz ((deriv-actfn (layer-actfn-kind layers (dec l))) y)
        dEdz (m/mult dEdy dydz)]
    dEdz))

(defmethod layer-error-deriv-wrt-weights
  :fully-connected
  [layers l dEdz-out ax _]
  (let [x    (:a ax)
        dEdw (m/* (m/t dEdz-out) (u/with-bias x))]
    dEdw))

