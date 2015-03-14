(ns synaptic.conv-test
  (:require [clojure.test :refer :all]
            [synaptic.conv :refer :all]
            [clatrix.core :as m])
  (:import  [org.jblas DoubleMatrix]))


(deftest test-pooling
  (testing "pooling-activities-and-indices"
    (let [x (m/matrix [[3 4 4 0  4 1 1 1  0 2 0 0   4 3 2 0  3 4 1 1  3 4 2 0]
                       [0 0 0 2  3 3 1 0  0 3 0 1   4 0 2 2  3 1 0 0  4 2 1 0]
                       [3 4 2 0  3 2 2 3  0 2 3 4   1 1 1 0  1 3 0 0  1 0 2 3]
                       [3 3 2 1  1 0 1 3  2 1 0 3   4 3 0 2  2 4 3 0  3 3 3 4]])
          pooling-result (pooling-activities-and-indices x 1 :max 2 3 2 2)]
      (is (= [[4.0 4.0 4.0 2.0]
              [3.0 2.0 4.0 2.0]
              [4.0 4.0 3.0 3.0]
              [3.0 3.0 4.0 4.0]]
             (m/dense (:a pooling-result)))))))

