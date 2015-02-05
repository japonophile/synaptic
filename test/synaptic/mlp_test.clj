(ns synaptic.mlp-test
  (:require [clojure.test :refer :all]
            [synaptic.mlp :refer :all]
            [synaptic.net :refer :all]
            [synaptic.util :refer :all]
            [clatrix.core :as m])
  (:import [synaptic.net Net]))


(defn mat-of-size? [mat m n]
  (and (m/matrix? mat)
       (= [m n] (m/size mat))))

(deftest test-weight-initialization
  (testing "init-all-weights for fully connected layers"
    (let [w (init-all-weights [{:type :input :n 5}
                               {:type :fully-connected :n 3 :act-fn :sigmoid}
                               {:type :fully-connected :n 2 :act-fn :sigmoid}])]
      (is (vector? w))
      (is (= 2 (count w)))
      (let [w1 (first w)
            w2 (second w)]
        (is (mat-of-size? w1 3 6))
        (is (mat-of-size? w2 2 4))
        (let [elems1 (apply concat (m/dense w1))
              elems2 (apply concat (m/dense w2))
              min1 (apply min elems1)
              max1 (apply max elems1)
              min2 (apply min elems2)
              max2 (apply max elems2)]
          (is (< 0 max1 (* 100 0.2)))
          (is (> 0 min1 (- (* 100 0.2))))
          (is (< 0 max2 (* 100 0.3)))
          (is (> 0 min2 (- (* 100 0.3)))))))
    (is (thrown? java.lang.AssertionError
                 (init-all-weights [])))
    (is (thrown? java.lang.AssertionError
                 (init-all-weights [{:type :input :n 5}])))))

(deftest test-neural-network
  (testing "mlp"
    (let [nn @(mlp [3 10 1] :sigmoid)]
      (is (= Net (type nn)))
      (let [w (:weights nn)
            l (-> nn :arch :layers)]
        (is (vector? w))
        (is (= 2 (count w)))
        (is (vector? l))
        (is (= 3 (count l)))
        (is (nil? (:act-fn (nth l 0))))
        (is (= :sigmoid (:act-fn (nth l 1))))
        (is (= :sigmoid (:act-fn (nth l 2))))))
    (let [nn @(mlp [6 4 4 2] :linear)]
      (is (= Net (type nn)))
      (let [w (:weights nn)
            l (-> nn :arch :layers)]
        (is (vector? w))
        (is (= 3 (count w)))
        (is (vector? l))
        (is (= 4 (count l)))
        (is (nil? (:act-fn (nth l 0))))
        (is (= :linear (:act-fn (nth l 1))))
        (is (= :linear (:act-fn (nth l 2))))
        (is (= :linear (:act-fn (nth l 3))))))))

(deftest test-network-outputs
  (testing "layer-activities :fully-connected"
    (let [w (m/matrix [[1 2 3][4 5 6]])
          x (m/matrix [[0 1][1 1][1 0][0 0]])]
      (is (= [[4.0 10.0] [6.0 15.0] [3.0 9.0] [1.0 4.0]]
             (m/dense (:a (layer-activities
                            [{:type :input :n 2}
                             {:type :fully-connected :n 2 :act-fn :linear}] 1 w x)))))
      (is (= [[0.9820137900379085 0.9999546021312976]
              [0.9975273768433653 0.999999694097773]
              [0.9525741268224334 0.9998766054240137]
              [0.7310585786300049 0.9820137900379085]]
             (m/dense (:a (layer-activities
                            [{:type :input :n 2}
                             {:type :fully-connected :n 2
                              :act-fn :sigmoid}] 1 w x))))))))

