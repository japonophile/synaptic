(ns synaptic.net-test
  (:require [clojure.test :refer :all]
            [synaptic.net :refer :all]
            [synaptic.mlp :refer [mlp]]
            [synaptic.util :refer :all]
            [clatrix.core :as m]))


(deftest test-layer-units
  (let [layers1 (instrument-layers
                  [{:type :input :fieldsize [3 28 28]}
                   {:type :convolution :feature-map {:size [9 9] :k 6}
                    :act-fn :hyperbolic-tangent}
                   {:type :convolution :feature-map {:size [5 5] :k 16}
                    :act-fn :hyperbolic-tangent
                    :pool {:kind :max :size [2 2]}}
                   {:type :fully-connected :n 10 :act-fn :softmax}])
        layers2 (instrument-layers
                  [{:type :input :fieldsize [24 24]}
                   {:type :fully-connected :n 1 :act-fn :sigmoid}])]
    (testing "layer-units should return the nb of outputs of the input layer"
      (is (= 2352 (layer-units layers1 0))))
    (testing "layer-units should return the nb of outputs of the input layer"
      (is (= 576  (layer-units layers2 0))))
    (testing "layer-units should return the nb of outputs of a convolution layer"
      (is (= 2400 (layer-units layers1 1))))
    (testing "layer-units should return the nb of outputs of a pooled conv layer"
      (is (= 1024 (layer-units layers1 2))))
    (testing "layer-units should return the nb of outputs of a fully-connected layer"
      (is (= 10   (layer-units layers1 3))))))

(deftest test-activation-functions
  (testing "binary-threshold"
    (is (m-quasi-equal? [[1 0 1 0]]
                        (binary-threshold (m/matrix [[0 -0.001 0.001 -1000]])))))
  (testing "bipolar-threshold"
    (is (m-quasi-equal? [[1 -1 1 -1]]
                        (bipolar-threshold (m/matrix [[0 -0.001 0.001 -1000]])))))
  (testing "sigmoid"
    (is (m-quasi-equal? [[0.25 0.2]]
                        (sigmoid (m/matrix [[(Math/log 1/3) (Math/log 1/4)]])))))
  (testing "hyperbolic-tangent"
    (is (m-quasi-equal? [[0.7615942 0.9640276 0.9950548]]
                        (hyperbolic-tangent (m/matrix [[1 2 3]])))))
  (testing "sample-with-p"
    (rand-set! 1)  ; 0.7308782 0.100473166 0.4100808 ...
    (is (m-quasi-equal? [[1 0 1]]
                        (sample-with-p (m/matrix [[0.74 0.09 0.58]])))))
  (testing "binary-stochastic"
    (rand-set! 1)  ; 0.7308782 0.100473166 0.4100808 ...
    (is (m-quasi-equal? [[0 1 0]]
                        (binary-stochastic (m/matrix [[0.5 -2.0 -0.5]])))))
  (testing "binary-prob"
    (is (m-quasi-equal? [[1 0 0]]
                        (binary-prob (m/matrix [[0.5 -2.0 -0.5]])))))
  (testing "softmax"
    (is (m-quasi-equal? [[0.0900306 0.2447285 0.6652410]
                         [0.6652410 0.2447285 0.0900306]]
                        (softmax (m/matrix [[1 2 3][2 1 0]]))))))

(deftest test-network-outputs
  (testing "valid-inputs?"
    (let [nn @(mlp [3 4 2 1] :binary-prob)]
      (is (false? (valid-inputs? nn (m/matrix []))))
      (is (false? (valid-inputs? nn nil)))))
  (testing "net-activities"
    (let [nn  @(mlp [3 2 1] :binary-threshold)
          x   (m/matrix [[0 1 0][1 0 1][1 1 0][1 0 0][1 1 1]])
          act (net-activities nn x)]
      (is (vector? act))
      (is (= 2 (count act)))
      (is (every? #(m/matrix? (:a %)) act))
      (is (every? #(= 5 (first (m/size (:a %)))) act))
      (is (= 2 (second (m/size (:a (first act))))))
      (is (= 1 (second (m/size (:a (second act)))))))))

