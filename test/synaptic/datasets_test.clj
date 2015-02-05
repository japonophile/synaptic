(ns synaptic.datasets-test
  (:require [clojure.test :refer :all]
            [synaptic.datasets :refer :all]
            [synaptic.util :refer :all]
            [clatrix.core :as m])
  (:import [synaptic.datasets DataSet TrainingSet]))


(deftest test-create-training-set
  (testing "shuffle-vecs"
    (let [v1 ["1" "2" "3" "4" "5" "6"]
          v2 [1 2 3 4 5 6]
          [shv1 shv2] (shuffle-vecs v1 v2)]
      (is (every? true? (map #(= (Integer/parseInt %1) %2) shv1 shv2)))))
  (testing "partition-vecs"
    (let [[batchsmp batchlb] (partition-vecs 3 [[0 1][1 1][1 0][0 0][1 1][0 1]
                                                [1 1][0 0][0 1][1 1][1 0]]
                                               [[0][1][1][0][1][0][0][1][1][0][1]])]
      (is (= 4 (count batchsmp) (count batchlb)))
      (is (= 3 (count (first batchsmp)) (count (first batchlb))))
      (is (= 2 (count (last batchsmp)) (count (last batchlb)))))
    (let [[batchsmp batchlb] (partition-vecs nil [[0 1][1 1][1 0][0 0][1 1][0 1]
                                                  [1 1][0 0][0 1][1 1][1 0]]
                                                 [[0][1][1][0][1][0][0][1][1][0][1]])]
      (is (= 1 (count batchsmp) (count batchlb)))
      (is (= 11 (count (first batchsmp)) (count (first batchlb))))))
  (testing "training-set"
    (let [ts (training-set [[1 0 1][0 1 1][0 0 1][1 1 0][0 0 0]
                            [0 1 1][1 1 0][1 0 1][0 1 0]]
                           ["b" "a" "a" "b" "a" "b" "b" "a" "b"] {:batch 2})]
      (is (= TrainingSet (type ts)))
      (let [bs   (:batches ts)
            vs   (:valid ts)
            ulbs (-> ts :header :labels)]
        (is (vector? bs))
        (is (= 5 (count bs)))
        (is (every? #(= DataSet (type %)) bs))
        (is (nil? vs))
        (is (= ["a" "b"] ulbs))
        (let [x (:x (first bs))
              y (:y (first bs))]
          (is (m/matrix? x))
          (is (m/matrix? y))
          (is (= [2 3] (m/size x)))
          (is (= [2 2] (m/size y))))))
    (let [ts (training-set [[1 0][0 0][0 1][1 0][0 1][1 1][1 0][0 1][1 1][0 0]]
                           ["2" "0" "1" "2" "1" "3" "2" "1" "3" "0"]
                           {:nvalid 3})]
      (is (= TrainingSet (type ts)))
      (let [bs   (:batches ts)
            vs   (:valid ts)
            ulbs (-> ts :header :labels)]
        (is (vector? bs))
        (is (= 1 (count bs)))
        (is (= DataSet (type (first bs))))
        (is (= DataSet (type vs)))
        (is (= ["0" "1" "2" "3"] ulbs))
        (let [x (:x (first bs))
              y (:y (first bs))]
          (is (m/matrix? x))
          (is (m/matrix? y))
          (is (= [7 2] (m/size x)))
          (is (= [7 4] (m/size y))))
        (let [x (:x vs)
              y (:y vs)]
          (is (m/matrix? x))
          (is (m/matrix? y))
          (is (= [3 2] (m/size x)))
          (is (= [3 4] (m/size y))))))
    (let [smp [[1 0 1 0][0 1 0 1][0 0 1 1][1 1 0 1][0 0 1 1]]
          ts  (training-set smp ["+" "-" "-" "+" "+"] {:online true :rand false})]
      (is (= TrainingSet (type ts)))
      (let [bs   (:batches ts)
            vs   (:valid ts)
            ulbs (-> ts :header :labels)]
        (is (vector? bs))
        (is (= 5 (count bs)))
        (is (= DataSet (type (first bs))))
        (is (nil? vs))
        (is (= ["+" "-"] ulbs))
        (is (every? true? (map #(= [(map double %1)]
                                   (m/dense (:x %2))) smp bs)))))))

