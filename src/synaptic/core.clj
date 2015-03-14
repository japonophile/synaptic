(ns
  ^{:doc "Neural networks in Clojure"
    :author "Antoine Choppin"}
  synaptic.core
  (:require [clatrix.core :as m]
            [synaptic.net :as n]
            [synaptic.mlp :as mlp]
            [synaptic.convolution :as c]
            [synaptic.datasets :as d]
            [synaptic.training :as t]
            [synaptic.util :as u])
  (:gen-class))

(set! *warn-on-reflection* true)

(def neural-net        n/neural-net)
(def save-neural-net   n/save-neural-net)
(def load-neural-net   n/load-neural-net)
(def estimate          n/estimate)

(def mlp               mlp/mlp)

(def save-training-set d/save-training-set)
(def load-training-set d/load-training-set)
(def training-set      d/training-set)
(def test-set          d/test-set)

(def training          t/training)
(def train             t/train)
(def stop-training     t/stop-training)

