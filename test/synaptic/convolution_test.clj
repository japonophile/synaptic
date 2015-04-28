(ns synaptic.convolution-test
  (:require [clojure.test :refer :all]
            [synaptic.net :refer :all]
            [synaptic.datasets :refer :all]
            [synaptic.training :refer :all]
            [synaptic.convolution :refer :all]
            [synaptic.util :refer :all]
            [clatrix.core :as m])
  (:import  [synaptic.net Net]
            [synaptic.datasets DataSet]
            [org.jblas DoubleMatrix]))


(deftest test-weight-initialization
  (testing "init-all-weights for convolution layers"
    (let [layers (instrument-layers
                   [{:type :input :fieldsize [3 28 28]}
                    {:type :convolution :feature-map {:size [9 9] :k 6}
                     :act-fn :hyperbolic-tangent}
                    {:type :convolution :feature-map {:size [5 5] :k 16}
                     :act-fn :hyperbolic-tangent
                     :pool {:kind :max :size [2 2]}}
                    {:type :fully-connected :n 10 :act-fn :softmax}])
          w (init-all-weights layers)]
      (is (= 3 (count w)))
      (is (= [ 6  244] (m/size (nth w 0))))
      (is (= [16  151] (m/size (nth w 1))))
      (is (= [10 1025] (m/size (nth w 2)))))))

(deftest test-network-outputs
  (testing "layer-activites :convolution"
    (let [w (m/matrix [[0 1 0 1 2]   ; featuremap #1
                       [0 2 2 1 1]   ; featuremap #2
                       [0 0 1 0 1]]) ; featuremap #3
          x (m/matrix [[1 0 1 1 
                        0 1 1 1 
                        0 1 0 0 
                        1 0 0 0]
                       [1 0 0 0 
                        1 1 0 1 
                        0 1 1 1 
                        0 1 1 1]
                       [0 1 1 1 
                        0 1 1 0 
                        0 0 1 0 
                        1 1 0 1]
                       [0 0 0 1 
                        0 0 1 1 
                        0 1 1 1 
                        1 0 1 0]])
          act-conv (layer-activities
                     (instrument-layers
                       [{:type :input :fieldsize [4 4]}
                        {:type :convolution :feature-map {:size [2 2] :k 3}
                         :act-fn :linear}]) 1 w x)
          act-pool (layer-activities
                     (instrument-layers
                       [{:type :input :fieldsize [4 4]}
                        {:type :convolution :feature-map {:size [2 2] :k 3}
                         :act-fn :linear :pool {:kind :max :size [3 3]}}])
                     1 w x)]
      (is (= [[3.0 3.0 4.0 2.0 2.0 1.0 1.0 1.0 0.0  ; 1 0 1 2 (featuremap #1)
               3.0 4.0 6.0 3.0 5.0 4.0 3.0 2.0 0.0  ; 2 2 1 1 (featuremap #2)
               1.0 2.0 2.0 2.0 1.0 1.0 1.0 0.0 0.0] ; 0 1 0 1 (featuremap #3)
              [4.0 1.0 2.0 3.0 4.0 3.0 2.0 4.0 4.0  ; 1 0 1 2
               4.0 1.0 1.0 5.0 4.0 4.0 3.0 6.0 6.0  ; 2 2 1 1
               1.0 0.0 1.0 2.0 1.0 2.0 2.0 2.0 2.0] ; 0 1 0 1
              [2.0 4.0 2.0 0.0 3.0 2.0 3.0 1.0 3.0  ; 1 0 1 2
               3.0 6.0 5.0 2.0 5.0 3.0 2.0 3.0 3.0  ; 2 2 1 1
               2.0 2.0 1.0 1.0 2.0 0.0 1.0 1.0 1.0] ; 0 1 0 1
              [0.0 2.0 3.0 2.0 3.0 4.0 1.0 3.0 2.0  ; 1 0 1 2
               0.0 1.0 4.0 1.0 4.0 6.0 3.0 5.0 5.0  ; 2 2 1 1
               0.0 1.0 2.0 1.0 2.0 2.0 1.0 2.0 1.0]]; 0 1 0 1
             (m/dense (:a act-conv))))
      (is (= [[4.0 6.0 2.0]
              [4.0 6.0 2.0]
              [4.0 6.0 2.0]
              [4.0 6.0 2.0]]
             (m/dense (:a act-pool))))
      ; note: when the max value is found multiple times, the index of the max
      ;       is the index of the *first* occurrence of that max value.
      (is (= [[2.0 11.0 19.0]
              [0.0 16.0 21.0]
              [1.0 10.0 18.0]
              [5.0 14.0 20.0]]
             (m/dense (:i act-pool)))))))

(defn close-enough?
  [mat1 mat2]
  (let [epsilon 0.001]
    (every? true?
            (map (fn [e1 e2]
                   (every? #(< % epsilon) (flatten (m/abs (m/- e1 e2)))))
                 mat1 mat2))))

(defn single-gradient-approx!
  "Compute an approximation of the gradient of the error for a given dataset, 
  around the current network weights, for a single weight.
  Note: This function uses clatrix `set` thereby temporary altering the weights 
  of nntemp (these weights are assumed to be a copy of the weights wij on which 
  this function is called)."
  [^Net nntemp dset l i j wij]
  (let [epsilon 1e-4
        wtemp   (nth (:weights nntemp) l)
        _       (m/mset! wtemp i j (+ wij epsilon))
        E1      (net-error-dataset nntemp dset)
        _       (m/mset! wtemp i j (- wij epsilon))
        E2      (net-error-dataset nntemp dset)
        _       (m/mset! wtemp i j wij)]
    (double (/ (- E1 E2) (* 2 epsilon)))))

(defn gradient-approx
  "Compute an approximation of the gradient of the error for a given dataset, 
  around the current network weights."
  [^Net nn ^DataSet dset]
  (let [ws    (:weights nn)
        [x y] [(:x dset) (:y dset)]]
    (vec (for [l (range (count ws))]
      (let [w      (nth ws l)
            nntemp (assoc nn :weights (assoc ws l (m/matrix w)))]
        (m/map-indexed (partial single-gradient-approx! nntemp dset l) w))))))

(deftest test-convolution-backprop
  (testing "error-derivatives with convolution layer: dEdws size"
    (let [_  (rand-set! 1)
          nn @(neural-net [{:type :input :fieldsize [3 28 28]}
                           {:type :convolution :feature-map {:size [9 9] :k 6}
                            :act-fn :hyperbolic-tangent}
                           {:type :fully-connected :n 10 :act-fn :softmax}]
                          (training :backprop))
          ds (DataSet. (m/rand 20 (* 3 28 28)) (m/rand 20 10))
          dEdws        (error-derivatives nn ds)]
      (is (= [[6 244] [10 2401]] (mapv m/size dEdws)))))
  (testing "error-derivatives with convolution layer: gradient checking"
    (let [_  (rand-set! 1)
          nn @(neural-net [{:type :input :fieldsize [3 8 8]}
                           {:type :convolution :feature-map {:size [5 5] :k 2}
                            :act-fn :hyperbolic-tangent}
                           {:type :fully-connected :n 1 :act-fn :sigmoid}]
                          (training :backprop))
          ds (DataSet. (m/rand 10 (* 3 8 8))
                       (m/matrix (map #(if (> % 0.5) 1.0 0.0) (m/rand 10 1))))
          dEdws        (error-derivatives nn ds)
          dEdws-approx (gradient-approx nn ds)]
      (is (= [[2 76] [1 33]] (mapv m/size (:weights nn))))
      (is (= [[2 76] [1 33]] (mapv m/size dEdws)))
      (is (= (count dEdws) (count dEdws-approx) 2))
      (is (close-enough? dEdws dEdws-approx))))
  (testing "error backpropagation with convolution layer: gradient checking"
    (let [_  (rand-set! 1)
          nn @(neural-net [{:type :input :fieldsize [3 10 10]}
                           {:type :convolution :feature-map {:size [5 5] :k 2}
                            :act-fn :hyperbolic-tangent}
                           {:type :convolution :feature-map {:size [3 3] :k 1}
                            :act-fn :hyperbolic-tangent}
                           {:type :fully-connected :n 1 :act-fn :sigmoid}]
                          (training :backprop))
          ds (DataSet. (m/rand 10 (* 3 10 10))
                       (m/matrix (map #(if (> % 0.5) 1.0 0.0) (m/rand 10 1))))
          dEdws        (error-derivatives nn ds)
          dEdws-approx (gradient-approx nn ds)]
      (is (= [[2 76] [1 19] [1 17]] (mapv m/size (:weights nn))))
      (is (= [[2 76] [1 19] [1 17]] (mapv m/size dEdws)))
      (is (= (count dEdws) (count dEdws-approx) 3))
      (is (close-enough? dEdws dEdws-approx))))
  (testing "error-derivatives with pooling convolution layer: gradient checking"
    (let [_  (rand-set! 1) ; this is not doing anything right now, because using
                           ; clatrix m/rand and not our own.  keeping it for now
                           ; to try generating new failure modes
          nn @(neural-net [{:type :input :fieldsize [3 8 8]}
                           {:type :convolution :feature-map {:size [5 5] :k 2}
                            :act-fn :hyperbolic-tangent
                            :pool {:kind :max :size [2 2]}}
                           {:type :fully-connected :n 1 :act-fn :sigmoid}]
                          (training :backprop))
          ds (DataSet. (m/rand 10 (* 3 8 8))
                       (m/matrix (map #(if (> % 0.5) 1.0 0.0) (m/rand 10 1))))
          dEdws        (error-derivatives nn ds)
          dEdws-approx (gradient-approx nn ds)]
      (is (= [[2 76] [1 9]] (mapv m/size (:weights nn))))
      (is (= [[2 76] [1 9]] (mapv m/size dEdws)))
      (is (= (count dEdws) (count dEdws-approx) 2))
      (is (close-enough? dEdws dEdws-approx))))
  (testing "error-derivatives with 2 pooling convolution layers"
    (let [nn @(neural-net [{:type :input :fieldsize [3 14 14]}
                           {:type :convolution :feature-map {:size [5 5] :k 4}
                            :act-fn :hyperbolic-tangent
                            :pool {:kind :max :size [2 2]}}
                           {:type :convolution :feature-map {:size [2 2] :k 2}
                            :act-fn :hyperbolic-tangent
                            :pool {:kind :max :size [2 2]}}
                           {:type :fully-connected :n 1 :act-fn :sigmoid}]
                          (training :backprop))
          ds (DataSet. (m/rand 10 (* 3 14 14))
                       (m/matrix (map #(if (> % 0.5) 1.0 0.0) (m/rand 10 1))))
          dEdws        (error-derivatives nn ds)
          dEdws-approx (gradient-approx nn ds)]
      (is (= [[4 76] [2 17] [1 9]] (mapv m/size (:weights nn))))
      (is (= [[4 76] [2 17] [1 9]] (mapv m/size dEdws)))
      (is (= (count dEdws) (count dEdws-approx) 3))
      (is (close-enough? dEdws dEdws-approx)))))

