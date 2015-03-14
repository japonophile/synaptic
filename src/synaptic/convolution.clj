(ns
  ^{:doc "synaptic - convolution layers"
    :author "Antoine Choppin"}
  synaptic.convolution
  (:require [clatrix.core :as m]
            [synaptic.conv :refer :all]
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

(defn convolution-layer-dimensions
  "Returns (k,w,h) input, output and weights of a convolution layer."
  [layers l]
  {:pre [(= :convolution (:type (nth layers l))) (> l 0)]}
  (let [[kin win hin]   (:fieldsize (nth layers (dec l)))
        layer           (nth layers l)
        [  w    h   ]   (:size (:feature-map layer))
        [k wout hout]   (:fieldsize layer)
        [  wc   hc  ]   (if-let [pool (:pool layer)]
                          (mapv * [wout hout] (:size pool))
                          [wout hout])]
    [kin win hin k wc hc wout hout w h]))


; Network outputs

(defn convolution-activities
  "Perform convolution of weights on x with given activation function and 
  precomputed convolution indices."
  [x weights actfn kin win hin k w h]
  (actfn (convolute x weights kin win hin k w h)))

(defmethod layer-activities
  :convolution
  [layers l weights x]
  (let [[kin win hin k wc hc wout hout w h] (convolution-layer-dimensions layers l)
        actfn      (layer-actfn layers l)
        convoluted (convolution-activities x weights actfn kin win hin k w h)]
    (if-let [pool (:pool (nth layers l))]
      (let [[wpool hpool] (:size pool)]
        (pooling-activities-and-indices convoluted k (:kind pool)
                                        wpool hpool wout hout))
      {:a convoluted})))


; Training

(defn assign-row-elements-by-index
  "Create a sparse matrix by assigning all elements of 'a' to the output matrix
  according to the indices specified by 'i'.  Each sample corresponds to a row
  of 'a' and of the output matrix.
  This is like the reverse operation of pooling."
  [kwhc as is]
  (m/matrix
    (map (fn [irow arow]
           (reduce (fn [v [i a]] (assoc v (int i) a))
                   (vec (repeat kwhc 0))
                   (map vector irow arow)))
         (m/dense is) (m/dense as))))

(defmethod prev-layer-error-deriv-wrt-logit
  :convolution
  [layers l dEdz-out ay weights]
  (let [[kin win hin k wc hc wout hout w h] (convolution-layer-dimensions layers l)
        y    (:a ay)
        dEdy (backprop-dEdz dEdz-out weights kin win hin k wc hc)
        dydz ((deriv-actfn (layer-actfn-kind layers (dec l))) y)
        dEdz (m/mult dEdy dydz)]
    dEdz))

(defmethod layer-error-deriv-wrt-weights
  :convolution
  [layers l dEdz-out ax ay]
  (let [[kin win hin k wc hc wout hout w h] (convolution-layer-dimensions layers l)
        x        (:a ax)
        indices  (:i ay)
        dEdz-out (if (:pool (nth layers l))
                   (assign-row-elements-by-index (* k wc hc) dEdz-out indices)
                   dEdz-out)
        dEdw     (sum-contributions x dEdz-out kin win hin wc hc)]
    dEdw))

