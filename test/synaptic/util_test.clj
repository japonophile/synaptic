(ns synaptic.util-test
  (:require [clojure.test :refer :all]
            [clatrix.core :as m]
            [synaptic.util :refer :all]))

(deftest test-random-number-generation
  (testing "random and nrand should return consistent values after setting the
           random generator seed to the same value"
    (rand-set! 1)
    (let [[r1 r2 r3] [(random) (random) (random)]
          [r4 r5 r6] [(nrand) (nrand) (nrand)]]
      (rand-set! 1)
      (is (= [r1 r2 r3] [(random) (random) (random)]))
      (is (= [r4 r5 r6] [(nrand) (nrand) (nrand)])))))

(deftest test-bit-array-manipulation
  (testing "pack should return an vector of short ints, each of which holds
           8 bits of data"
    (is (= [192 192] (pack [1 1 0 0 0 0 0 0
                            1 1 0 0 0 0 0 0])))
    (is (= [170 85]  (pack [1 0 1 0 1 0 1 0
                            0 1 0 1 0 1 0 1])))
    (is (= [255 80]  (pack [1 1 1 1 1 1 1 1
                            0 1 0 1])))
    (is (= [0]       (pack [0])))
    (is (= [128]     (pack [1])))
    (is (= []        (pack [])))
    (is (= [230 23
            27  46]  (pack [1 1 1 0 0 1 1 0
                            0 0 0 1 0 1 1 1
                            0 0 0 1 1 0 1 1
                            0 0 1 0 1 1 1 0])))
    (is (= [25  232
            228 209] (pack [0 0 0 1 1 0 0 1
                            1 1 1 0 1 0 0 0
                            1 1 1 0 0 1 0 0
                            1 1 0 1 0 0 0 1]))))
  (testing "unpack should return a vector of 0 and 1, corresponding to the bits
           of each int value of the input vector"
    (is (= [1 1 0 0 0 0 0 0
            1 1 0 0 0 0 0 0] (unpack [192 192])))
    (is (= [1 0 1 0 1 0 1 0
            0 1 0 1 0 1 0 1] (unpack [170 85])))
    (is (= [1 1 1 1 1 1 1 1
            0 1 0 1 0 0 0 0] (unpack [255 80])))
    (is (= [0 0 0 0 0 0 0 0] (unpack [0])))
    (is (= [1 0 0 0 0 0 0 0] (unpack [128])))))
  (testing "imgint2bits should produce a vector of 1 and 0 from an int vector
           representing an image of a given size (it should get rid of padding bits)"
    (is (= [1 1 0 0 1 1
            1 0 1 0 1 0
            1 0 1 0 1 0
            0 0 1 1 0 0
            1 1 0 0 1 1
            0 1 0 1 0 1
            0 1 0 1 0 1
            0 0 1 1 0 0]
           (imgint2bits [204 170 170 51 204 85 85 51] 6 8)))
    (is (= [1 1 0 0 1 1 0 0
            1 0 1 0 1 0 1 0
            1 0 1 0 1 0 1 0
            0 0 1 1 0 0 1 1
            1 1 0 0 1 1 0 0
            0 1 0 1 0 1 0 1
            0 1 0 1 0 1 0 1
            0 0 1 1 0 0 1 1]
           (imgint2bits [204 170 170 51 204 85 85 51] 8 8))))
  (testing "imgbits2int should produce a vector of int values by packing the
           0 and 1 values given as input, and padding them appropriately"
    (is (= [207 171 171 51 207 87 87 51]
           (imgbits2int [1 1 0 0 1 1
                         1 0 1 0 1 0
                         1 0 1 0 1 0
                         0 0 1 1 0 0
                         1 1 0 0 1 1
                         0 1 0 1 0 1
                         0 1 0 1 0 1
                         0 0 1 1 0 0] 6 8)))
    (is (= [204 170 170 51 204 85 85 51]
           (imgbits2int [1 1 0 0 1 1 0 0
                         1 0 1 0 1 0 1 0
                         1 0 1 0 1 0 1 0
                         0 0 1 1 0 0 1 1
                         1 1 0 0 1 1 0 0
                         0 1 0 1 0 1 0 1
                         0 1 0 1 0 1 0 1
                         0 0 1 1 0 0 1 1] 8 8))))

(deftest test-base64-encoding
  (testing "tobase64 should produce a Base-64 string encoding of the input"
    (is (= "ABCDEFGH" (tobase64 [0 16 131 16 81 135])))
           ;     0      1      2      3      4      5      6      7
           ;000000 000001 000010 000011 000100 000101 000110 000111
           ;00000000 00010000 10000011  00010000 01010001 10000111
           ;       0       16      131        16       81      135
    (is (= "ABCDEFE=" (tobase64 [0 16 131 16 81])))
    (is (= "ABCDEA==" (tobase64 [0 16 131 16]))))
  (testing "frombase64 should decode a Base-64 string into an int vector"
    (is (= [0 16 131 16 81 135] (frombase64 "ABCDEFGH")))
    (is (= [0 16 131 16 81]     (frombase64 "ABCDEFE=")))
    (is (= [0 16 131 16]        (frombase64 "ABCDEA==")))))

(deftest test-binary-encoding
  (testing "tobinary should return the vector of unique labels and all labels
           encoded to binary vectors"
    (is (= [[[0 0 0 1] [1 0 0 0] [0 1 0 0] [0 0 0 1] [0 0 1 0] [1 0 0 0] [0 0 1 0]]
            {[0 0 0 1] "8", [0 0 1 0] "3", [0 1 0 0] "2", [1 0 0 0] "1"}] 
           (tobinary ["8" "1" "2" "8" "3" "1" "3"]))))
  (testing "frombinary should decode each label to its original value, based
           on a vector of unique labels"
    (is (= ["8" "1" "2" "8" "3" "1" "3"]
           (frombinary ["1" "2" "3" "8"]
                       [[0 0 0 1] [1 0 0 0] [0 1 0 0] [0 0 0 1]
                        [0 0 1 0] [1 0 0 0] [0 0 1 0]])))))

(deftest test-continous-scaling
  (testing "tocontinous should return the vector of unique labels and all labels
           scaled to vectors with values in range 0 to 1, and a function to scale them back"
    (is (= [[0.4 0.6] [0.8 1.0] [0.0 0.2]] 
           (first (tocontinous [[1 2] [3 4] [-1 0]]))))
    (is (= (m/matrix [[-1 4]])
           ((second (tocontinous [[1 2] [3 4] [-1 0]])) [[0 1]])))))

(deftest test-data-manipulation
  (testing "unique should return a sorted vector of unique values"
    (is (= ["a" "b" "c" "d" "x" "y" "z"]
           (unique ["c" "c" "x" "a" "z" "b" "a" "c" "y" "z" "d"])))))

(deftest test-print-read-matrices
  (testing "print-method should print a matrix in EDN format"
    (is (= "#clatrix.core/Matrix [[1.0 2.0 3.0] [4.0 5.0 6.0]]"
           (pr-str (m/t (m/matrix [[1 4][2 5][3 6]]))))))
  (testing "read-string should read a matrix expressed in EDN format"
    (is (m-quasi-equal? [[1 0.1][0.9 9]]
                        (read-matrix [[1.0 0.1][0.9 9.0]])))))

(deftest test-matrix-utilities
  (testing "Two numbers should be equal if their absolute difference is less than eps"
    (is (quasi-equal? 1.0 1.0000009))
    (is (quasi-equal? 1.0 0.9999991))
    (is (not (quasi-equal? 1.0 1.0000011)))
    (is (not (quasi-equal? 1.0 0.999999))))
  (testing "Data and matrix are quasi-equal if their content is quasi-equal"
    (is (not (m-quasi-equal? [[1.0 2.0 3.0][4.0 5.0 6.0]]
                             (m/matrix [[1.0 2.0 3.1] [4.0 5.0 6.0]]))))
    (is (m-quasi-equal? [[1.0 2.0 3.0999991][4.0 5.0 6.0]]
                        (m/matrix [[1.0 2.0 3.1] [4.0 5.0 6.0]])))))

(deftest test-stat-utilities
  (testing "histogram"
    (is (= [1]           (:data (histogram [1] 1))))
    (is (= [1 2 1]       (:data (histogram [1 2 2 3] 3))))
    (is (= [1 0 2 3 4 1]
           (:data (histogram [-1.6 3.0 1.91 0.9 1.1 1.9 2.1 2.95 3.1 2.9 4.1] 6))))
    (is (= [1 0 0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0 0 1] (:data (histogram [1 2]))))))

