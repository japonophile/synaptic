(ns
  ^{:doc "Neural networks in Clojure"
    :author "Antoine Choppin"}
  synaptic.core
  (:require [clatrix.core :as m]
            [synaptic.net :as n]
            [synaptic.mlp :as p]
            [synaptic.convolution :as c]
            [synaptic.datasets :as d]
            [synaptic.training :as t]
            [synaptic.util :as u])
  (:gen-class))

(set! *warn-on-reflection* true)

(def neural-net        n/neural-net)
(alter-meta! #'neural-net merge
             (select-keys (meta #'n/neural-net)        [:doc :arglists]))
(def save-neural-net   n/save-neural-net)
(alter-meta! #'save-neural-net merge
             (select-keys (meta #'n/save-neural-net)   [:doc :arglists]))
(def load-neural-net   n/load-neural-net)
(alter-meta! #'load-neural-net merge
             (select-keys (meta #'n/load-neural-net)   [:doc :arglists]))
(def estimate          n/estimate)
(alter-meta! #'estimate merge
             (select-keys (meta #'n/estimate)          [:doc :arglists]))

(def mlp               p/mlp)
(alter-meta! #'mlp merge
             (select-keys (meta #'p/mlp)               [:doc :arglists]))

(def save-training-set d/save-training-set)
(alter-meta! #'save-training-set merge
             (select-keys (meta #'d/save-training-set) [:doc :arglists]))
(def load-training-set d/load-training-set)
(alter-meta! #'load-training-set merge
             (select-keys (meta #'d/load-training-set) [:doc :arglists]))
(def load-training-set-header d/load-training-set-header)
(alter-meta! #'load-training-set-header merge
             (select-keys (meta #'d/load-training-set-header) [:doc :arglists]))
(def training-set      d/training-set)
(alter-meta! #'training-set merge
             (select-keys (meta #'d/training-set)      [:doc :arglists]))
(def test-set          d/test-set)
(alter-meta! #'test-set merge
             (select-keys (meta #'d/test-set)          [:doc :arglists]))

(def training          t/training)
(alter-meta! #'training merge
             (select-keys (meta #'t/training)          [:doc :arglists]))
(def train             t/train)
(alter-meta! #'train merge
             (select-keys (meta #'t/train)             [:doc :arglists]))
(def stop-training     t/stop-training)
(alter-meta! #'stop-training merge
             (select-keys (meta #'t/stop-training)     [:doc :arglists]))

