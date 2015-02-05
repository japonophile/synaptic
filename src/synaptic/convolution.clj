(ns
  ^{:doc "synaptic - convolution layers"
    :author "Antoine Choppin"}
  synaptic.convolution
  (:require [clatrix.core :as m]
            [synaptic.net :refer :all]
            [synaptic.training :refer :all]
            [synaptic.util :as u])
  (:gen-class))


; Network architecture

(defmethod layer-units
  :convolution
  [layers l]
  {:pre [(< 0 l (dec (count layers)))]}
  (let [layer (nth layers l)]
    (apply * (:fieldsize layer))))

(defmethod layer-instrument
  :convolution
  [layers l]
  (let [layer (nth layers l)]
    (let [prev-fieldsize (rest (:fieldsize (nth layers (dec l))))
          [fmap-size k]  (map #(% (:feature-map layer)) [:size :k])
          fieldsize      (map #(- %1 (dec %2)) prev-fieldsize fmap-size)
          fieldsize      (if-let [pool (:pool layer)]
                           (map / fieldsize (:size pool))
                           fieldsize)]
      (assoc layer :fieldsize (into [k] fieldsize)))))


; Weight initialization

(defmethod init-weights
  :convolution
  [layers l]
  {:pre [(< 0 l (count layers))]}
  (let [layer (nth layers l)
        prev-k (-> (nth layers (dec l)) :fieldsize first)
        [fmap-size k] (map #(% (:feature-map layer)) [:size :k])
        nin (* prev-k (apply * fmap-size))]
    (m/matrix (for [i (range k)]
                (for [j (range (inc nin))]
                  (* (/ 1.0 nin) (u/nrand)))))))


; Network outputs

(defn convolution-indices
  "Returns an array of array of indices, one array for each 'patch' involved 
  in the convolution.
  Since patches are overlapping (from the nature of the convolution operation), 
  the same is included in multiple patches, so the same index will be present in 
  multiple arrays of indices.
  If the input has multiple channels (or feature maps, that is if kin > 1), 
  there will be kin arrays of indices per patch (instead of just 1)."
  [kin win hin w h & [order]]
  (let [fsizein (* win hin)
        indices (if (= :reversed order) (reverse (range fsizein)) (range fsizein))]
    (loop [indices indices, conv-indices []]
      (if (< (count indices) (+ w (* (dec h) win)))
        conv-indices
        (let [fmap-indices     (flatten (take h (partition w win indices)))
              all-fmap-indices (vec (flatten (for [i (range kin)
                                                   l fmap-indices]
                                               (+ (* i fsizein) l))))
              next-indices     (if (>= (mod (dec (count indices)) win) w)
                                 (rest indices)
                                 (nthrest indices w))]
          (recur next-indices (conj conv-indices all-fmap-indices)))))))

(defn pooling-indices
  "Returns an array of array of indices, one array per 'block' to be pooled.
  If the layer has multiple feature maps (that is if k > 1), blocks for other 
  feature maps are also generated."
  [k w h wpool hpool]
  {:pre [(= 0 (mod w wpool)) (= 0 (mod h hpool))]}
  (let [fsize           (* w h)
        indices         (range fsize)
        colsofblockrows (apply (partial map vector)
                               (partition (/ w wpool) (partition wpool indices)))
        colsofblocks    (map (partial map flatten)
                             (map (partial partition hpool) colsofblockrows))
        blocks          (apply concat (apply (partial map vector) colsofblocks))]
     (vec (apply concat (for [i (range k)]
                          (map (partial mapv (partial + (* i fsize))) blocks))))))

(defn pad-four-sides
  "Pad x with zeros on all 4 sides.  Note that x is a 3-dimensional array 
  flattened to 2 dimensions, where each row is a flattened 2D matrix 
  corresponding to a single sample.  So padding will in fact insert columns of 
  zeros not only at the left and right of the matrix, but also in the middle 
  (in between each row of the 2D matrices)."
  [x w h wout hout]
  {:post [(= (second (m/size %)) (* wout hout (/ (second (m/size x)) (* w h))))
          (= (first (m/size x)) (first (m/size %)))]}
  (let [n          (first (m/size x))
        k          (/ (second (m/size x)) (* w h))
        half-wpad  (/ (- wout w) 2)
        half-hpad  (/ (- hout h) 2)
        pad-updown (m/zeros n (+ (* half-hpad wout) half-wpad))
        pad-sides  (m/zeros n (* 2 half-wpad))]
    (apply m/hstack
      (apply concat
        (for [fmap-cols (partition (* w h) (m/cols x))]
          (concat
            [pad-updown]
            (apply concat
              (map vector (map (partial apply m/hstack) (partition w fmap-cols))
                          (conj (vec (repeat (dec h) pad-sides)) (m/zeros n 0))))
            [pad-updown]))))))

(defn convolute
  "Perform convolution of b onto a, given pre-computed convolution indices."
  [a b conv-indices]
  (let [n (first (m/size a))]
    (apply m/hstack   ; create a big matrix with all columns
      (apply concat   ; concat columns of all fieldmaps
        (apply (partial map vector) ; group columns of same fieldmap
          (for [indices conv-indices]
            (let [as-with-bias (apply m/hstack (m/ones n) (m/cols a indices))]
              (m/cols (m/* as-with-bias (m/t b))))))))))

(defn sum-convolutions
  "Perform convolution of dEdw onto x, given pre-computed convolution indices,
  but reshape x to sum on all samples and all elements of the output."
  [x dEdz conv-indices kin]
  (let [n               (first (m/size x))
        wout-hout       (/ (count (first conv-indices)) kin)
        n-wout-hout     (* n wout-hout)
        k               (/ (second (m/size dEdz)) wout-hout)
        dEdz-reshaped-t (m/t (m/matrix (m/reshape dEdz n-wout-hout k)))]
    (apply m/hstack   ; create a big matrix with all columns
      (cons
        (m/* dEdz-reshaped-t (m/ones n-wout-hout))
        (apply concat   ; concat columns of all fieldmaps
          (apply (partial map vector) ; group columns of same fieldmap
            (for [indices conv-indices]
              (let [xs          (apply m/hstack (m/cols x indices))
                    xs-reshaped (m/matrix (m/reshape xs n-wout-hout kin))]
                (m/cols (m/* dEdz-reshaped-t xs-reshaped))))))))))

(defn convolution-activities
  "Perform convolution of weights on x with given activation function and 
  precomputed convolution indices."
  [x weights actfn conv-indices]
  (actfn (convolute x weights conv-indices)))

(defn pooled-sample-indices
  "Return indices of a single data sample corresponding to the max elements."
  [sample pool-kind pool-indices]
  (if (= :max pool-kind) ; currently only support max
    ;(mapv #(apply max-key sample %) pool-indices)))
    (mapv (fn [is]
            (let [i  (apply max-key sample is)
                  m  (nth sample i)
                  vs (filter (fn [s]
                               (> 0.00001 (Math/abs ^double (double (- s m)))))
                             (map sample is))
                  n  (count vs)]
              (if (> n 1)
                (do
                  (printf "more than 1 (%d) max value %f\n" n m)
                  (prn vs)))
              i)) pool-indices)))

(defn select-row-elements-by-index
  "Select elements of each row of 'a' corresponding to the indices specified by 'i'."
  [as is]
  {:pre [(= (first (m/size as)) (first (m/size is)))]
   :post [= (m/size is) (m/size %)]}
  (m/matrix (map (fn [a i] (map #(a (int %)) i)) (m/dense as) (m/dense is))))

(defn assign-row-elements-by-index
  "Create a matrix by assigning elements of each row of 'a' to the indices specified 
  by 'i'."
  [n as is]
  (m/matrix
    (map (fn [a i]
           (reduce #(assoc %1 (int (first %2)) (second %2))
                   (vec (repeat n 0))
                   (map vector i a)))
         (m/dense as) (m/dense is))))

(defn pooling-activities-and-indices
  "Computed pooled activities on the result of the convolution, and corresponding 
  indices."
  [convoluted-x pool-kind pool-indices]
  {:pre [(= (second (m/size convoluted-x)) (count (apply concat pool-indices)))]}
  (let [pooled-smp-ind (m/matrix
                         (map #(pooled-sample-indices
                                 (first (m/dense %)) pool-kind pool-indices)
                              convoluted-x))]
    {:a (select-row-elements-by-index convoluted-x pooled-smp-ind)
     :i pooled-smp-ind}))

(defmethod layer-activities
  :convolution
  [layers l weights x]
  (let [[kin win hin] (:fieldsize (nth layers (dec l)))
        layer         (nth layers l)
        [    w   h  ] (:size (:feature-map layer))
        actfn         (layer-actfn layers l)
        conv-indices  (convolution-indices kin win hin w h)
        convoluted-x  (convolution-activities x weights actfn conv-indices)]
    (if-let [pool (:pool layer)]
      (let [[k wout hout] (:fieldsize layer)
            [wpool hpool] (:size pool)
            pool-indices  (pooling-indices k (* wout wpool) (* hout hpool) wpool hpool)]
        (pooling-activities-and-indices convoluted-x (:kind pool) pool-indices))
      {:a convoluted-x})))


; Training

(defn convolution-layer-dimensions
  [layers l]
  {:pre [(= :convolution (:type (nth layers l))) (> l 0)]}
  (let [[kin win hin]   (:fieldsize (nth layers (dec l)))
        layer           (nth layers l)
        [    w   h  ]   (:size (:feature-map layer))
        fieldsize       (:fieldsize layer)
        [k wout hout]   (if-let [pool (:pool layer)]
                          (mapv * fieldsize (into [1] (:size pool)))
                          fieldsize)]
    [kin win hin k wout hout w h]))

(defmethod prev-layer-error-deriv-wrt-logit
  :convolution
  [layers l dEdz-out ay weights]
  (let [y               (:a ay)
        [kin win hin k wout hout w h] (convolution-layer-dimensions layers l)
        dEdz-out-padded (pad-four-sides dEdz-out wout hout win hin)
        conv-indices    (convolution-indices k win hin w h :reversed)
        dEdy            (convolute dEdz-out (u/without-bias weights) conv-indices)
        dydz            ((deriv-actfn (layer-actfn-kind layers (dec l))) y)
        dEdz            (m/mult dEdy dydz)]
    dEdz))

(defmethod layer-error-deriv-wrt-weights
  :convolution
  [layers l dEdz-out ax ay]
  (let [x            (:a ax)
        [kin win hin k wout hout w h] (convolution-layer-dimensions layers l)
        dEdz-out     (if (:pool (nth layers l))
                       (assign-row-elements-by-index (* k wout hout) dEdz-out (:i ay))
                       dEdz-out)
        conv-indices (convolution-indices kin win hin wout hout)
        dEdw         (sum-convolutions x dEdz-out conv-indices kin)]
    dEdw))

