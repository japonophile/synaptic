(ns
  ^{:doc "Neural networks in Clojure"
    :author "Antoine Choppin"}
  synaptic.core
  (:require [clatrix.core :as m]
            [synaptic.util :as u]
            [bozo.core :as b])
  (:import  [java.io File FileWriter]
            [clatrix.core Matrix])
  (:gen-class))

(set! *warn-on-reflection* true)


(defrecord Arch [layers act-fn])
(defrecord Stats [epochs tr-err val-err])
(defrecord Training [algo params state stats])
(defrecord Net [arch weights training])

;  Weight initialization

(def init-eps 0.01)

(defn init-weights
  "Initialize weights of a single layer, comprising nout units,
  each with nin inputs."
  [nin nout]
  (m/matrix (for [i (range nout)]
              (for [j (range (inc nin))]
                (* init-eps (u/nrand))))))

(defn init-all-weights
  "Initialize weights of all layers."
  [layers]
  {:pre [(< 1 (count layers))]}
  (vec (for [i (range (dec (count layers)))]
         (init-weights (nth layers i) (nth layers (inc i))))))

; Activation functions

(defn linear
  "Linear activation function.
  Returns the logit."
  [z]
  z)

(defn binary-threshold
  "Binary threshold activation function.
  Returns 1 if the logit is >= 0, or 0 otherwise."
  [zs]
  (m/map (fn [z] (if (>= z 0) 1 0)) zs))

(defn bipolar-threshold
  "Binary threshold activation function.
  Returns 1 if the logit is >= 0, or -1 otherwise."
  [zs]
  (m/map (fn [z] (if (>= z 0) 1 -1)) zs))

(defn sigmoid
  "Sigmoid activation function.
  Computed as 1 / (1 + e^(-z))."
  [zs]
  (m/div 1.0 (m/+ 1.0 (m/exp (m/- zs)))))

(defn sample-with-p
  "Returns a random sample with probability p."
  [ps]
  (m/map (fn [p] (if (< (u/random) p) 1 0)) ps))

(defn binary-stochastic
  "Binary stochastic activation function.
  Returns a random sample with probability given by the logistic (sigmoid)
  of the logit."
  [zs]
  (sample-with-p (sigmoid zs)))

(defn binary-prob
  "Binary probability activation function.
  Returns 1 if the sigmoid of the logit is >= 0.5, or 0 otherwise."
  [zs]
  (m/map (fn [z] (if (< z 0.5) 0 1)) (sigmoid zs)))

(defn softmax
  "Softmax activation function.
  Returns e(z) / Sum[e(z)]."
  [zs]
  (let [k    (m/ncols zs)
        nzs  (m/- zs (m/* (m/matrix (mapv (partial apply max) (m/rows zs)))
                          (m/ones 1 k)))
        es   (m/exp nzs)
        sums (m/* (m/matrix (mapv (partial reduce +) es))
                  (m/ones 1 k))]
    (m/div es sums)))

(defn actfn-kind
  "Returns the kind of activation function (expressed as a keyword) for a network 
  at a given layer (or last layer)."
  [^Net nn & [l]]
  (let [actfn (-> nn :arch :act-fn)]
    (if (vector? actfn) (if l (nth actfn l) (last actfn)) actfn)))

(defn act-fn
  "Returns the activation function for a network at a given layer (or last layer)."
  [^Net nn & [l]]
  (let [kind (actfn-kind nn l)]
    (resolve (symbol "synaptic.core" (name kind)))))

; Constructor

(defn neural-net
  "Create a neural network, and returns an atom pointing to it.
  Takes as input a seq with the number of units in each layer,
  and the activation function of network units (as a symbol).
  
  Ex. (neural-net [10 5 1] :sigmoid})
    creates a network with 10 input units, 5 hidden units and 1 output unit,
    using the sigmoid activation function.
  
  It is possible to define an activation function per layer.
  Ex. (neural-net [10 20 10 5] [:sigmoid :sigmoid :softmax])
  (There is no activation function for the input layer).
  
  It is also possible to initialize the training for the network.
  Ex. (neural-net [100 20 50] :sigmoid (training :backprop))
  See (doc training) for details."
  [layers actfn & [training]]
  {:pre [(vector? layers)
         (or (keyword? actfn) (= (dec (count layers)) (count actfn)))]}
  (atom (Net. (Arch. layers actfn) (init-all-weights layers) training)))

; Network outputs (neuron activities, forward pass)

(defn activities
  "Compute activities of a layer of neurons, with the specified activation
  function act-fn, given the layer weights w and layer inputs x."
  [actfn w x]
  {:pre [(m/matrix? w) (m/matrix? x)
         (= (second (m/size w)) (inc (second (m/size x))))]}
  (actfn (m/* (u/with-bias x) (m/t w))))

(defn valid-inputs?
  "Returns true if x are valid inputs for the neural network, else otherwise."
  [^Net nn x]
  {:pre [(= Net (type nn))]}
  (and (m/matrix? x)
       (= (second (m/size (first (:weights nn))))
          (inc (second (m/size x))))))

(defn net-activities
  "Compute activities of all layers of a neural network, for the given inputs x."
  [^Net nn x]
  {:pre [(= Net (type nn)) (valid-inputs? nn x)]}
  (loop [i 0, x x, ws (:weights nn), as []]
    (if (first ws)
      (let [actfn (act-fn nn i)
            a (activities actfn (first ws) x)]
        (recur (inc i) a (rest ws) (conj as a)))
      as)))

; Training set

(defrecord DataSet [x y])
(defrecord TrainingSet [batches val-set uniquelabels])

(defn encode-sample
  "Encode a vector of 0 and 1 into a Base64 string."
  [[w h] bits]
  (u/tobase64 (u/imgbits2int bits w h)))

(defn decode-sample
  "Decode a Base64-encoded sample into a vector of 0 and 1."
  [[w h] sample]
  (u/imgint2bits (u/frombase64 sample) w h))

(defn load-labeled-samples
  "Load labeled samples from disk."
  [sname]
  (let [lbsmp (u/loaddata "labeledsamples" sname)
        fsize (-> lbsmp :header :fieldsize)]
    (assoc lbsmp :samples
                 (mapv (partial decode-sample fsize) (:samples lbsmp)))))

(defn shuffle-vecs
  "Randomly shuffle multiple vectors of same size in the same way."
  [& vectors]
  {:pre [(apply = (map count vectors))]}
  (let [idx (shuffle (range (count (first vectors))))]
    (mapv #(mapv (partial nth %) idx) vectors)))

(defn partition-vecs
  "Partition multiple vectors of the same size in chunks (batches) of the
  specified size (just returns the input vectors if batchsize is nil)."
  [batchsize & vectors]
  {:pre [(apply = (map count vectors))]}
  (if (and batchsize (> batchsize 0) (< batchsize (count (first vectors))))
    (mapv (comp vec (partial map vec) (partial partition-all batchsize)) vectors)
    (mapv vector vectors)))

(defn dataset
  "Create a dataset with x and y."
  [x y]
  {:pre [(= (count x) (count y))]}
  (DataSet. (m/matrix x) (m/matrix y)))

(defn merge-batches
  "Merge batches into a single dataset."
  [batches]
  (if (> (count batches) 1)
    (DataSet. (apply m/vstack (map :x batches)) (apply m/vstack (map :y batches)))
    (first batches)))

(defn training-set
  "Create a training set from samples and associated labels.
  The training set consists of one or more batches and optionally a validation set.
  It also has a map that will allow converting y's back to the original labels.
  
  Options:
    :nvalid size - size of the validation set (default is 0, i.e. no validation set)
    :batch size  - size of a mini-batch (default is the number of samples, after
                   having set apart the validation set)
    :online true - set this flag for online training (same as batch size = 1)
    :rand false  - unset this flag to keep original ordering (by default, samples
                   will be shuffled before partitioning)."
  [samples labels & [options]]
  {:pre [(= (count samples) (count labels))]}
  (let [batchsize  (if (:online options) 1 (:batch options))
        trainsize  (if (:nvalid options) (- (count samples) (:nvalid options)))
        randomize  (if (nil? (:rand options)) true (:rand options))
        [binlb uniquelb]    (u/tobinary labels)
        [smp lb]   (if randomize (shuffle-vecs samples binlb) [samples binlb])
        [trainsmp validsmp] (if trainsize (split-at trainsize smp) [smp nil])
        [trainlb  validlb]  (if trainsize (split-at trainsize lb) [lb nil])
        [batchsmp batchlb]  (partition-vecs batchsize trainsmp trainlb)
        trainsets  (mapv dataset batchsmp batchlb)
        validset   (if trainsize (dataset validsmp validlb))]
    (TrainingSet. trainsets validset uniquelb)))

; Training

(defn training
  "Initialize network training for a given algorithm with optional training 
  parameters.  Currently supported algorithms are :perceptron, :backprop,
  :rprop, :rmsprop and :lbfgs.

  Training parameters:
  {
    :cost-fn            :misclassification, :sum-of-squares, :cross-entropy
    :learning-rate {    (required for :backprop and :rmsprop)
        :epsilon        learning rate (default: 0.01)
        :adaptive       true for adaptive learning rate
        :ming / :maxg   min / max gain (reqd for adaptive lr, recom.: 0.01 / 10.0)
      }
    :momentum {
        :alpha          momentum factor (reqd if you specify momentum)
        :alpha-start    initial value of momentum factor (optional)
        :alpha-step     step to increase momentum factor (from alpha-start to alpha)
        :nesterov       true to use Nesterov's momentum method
      }
    :rprop {
        :mins / :maxs   min / max step size (reqd for rprop, default 1e-6 / 50.0)
    }
    :rmsprop {
        :alpha          factor to compute moving avg of square gradient (def: 0.9)
    }
  }"
  [algo & [params]]
  {:pre [(#{:perceptron :backprop :rprop :rmsprop :lbfgs} algo)]}
  (let [default-params
        (case algo
          :perceptron {:cost-fn :misclassification}
          :backprop   {:cost-fn :cross-entropy
                       :learning-rate {:epsilon 0.01}}
          :rprop      {:cost-fn :cross-entropy
                       :rprop {:mins 0.000001 :maxs 50.0}}
          :rmsprop    {:cost-fn :cross-entropy
                       :rmsprop {:alpha 0.9}
                       :learning-rate {:epsilon 0.01}}
          :lbfgs      {:cost-fn :cross-entropy})]
    (Training. algo (merge default-params params) {} nil)))

; Learning

(defmulti deltaw
  "Compute delta of weights of the neural network, based on a dataset.
  Returns the (possibly updated) network and a vector of deltaw matrices 
  (one per weight matrix).
  What exactly this method does depends on the type of training algorithm."
  (fn [nn _] (if (-> nn :training :params :momentum :nesterov) :nesterov
               (-> nn :training :algo))))

(defn update-weights
  "Update neural network weights using computed delta weight."
  [weights deltaw]
  (vec (pmap (fn [w dw] (m/+ w dw)) weights deltaw)))

; Perceptron learning

(defn perceptron-error
  "Computes the error between perceptron output and target."
  [outputs targets]
  {:pre [(m/matrix? outputs) (m/matrix? targets)
         (= (m/size outputs) (m/size targets))]}
  (m/map-indexed
    (fn [i j o]
      (let [t (m/mget targets i j)]
        (cond
          (< o t)  1
          (> o t) -1
          :default 0))) outputs))

(defmethod deltaw
  :perceptron
  [^Net nn ^DataSet dset]
  {:pre [(= 1 (count (:weights nn)))]}  ; single layer
  (let [[x t] [(:x dset) (:y dset)]
        m     (first (m/size x))
        y     (last (net-activities nn x))
        err   (perceptron-error y t)]
    [nn [(m/* (m/t err) (u/with-bias x))]]))

; Backpropagation learning

; - Error derivatives

(defn deriv-actfn
  "Returns the derivative of an activation function (expressed as a keyword)."
  [actkind]
  (case actkind
    :softmax (fn [ys] (m/mult ys (m/- 1.0 ys)))
    :sigmoid (fn [ys] (m/mult ys (m/- 1.0 ys)))))

(defn output-layer-error-deriv
  "Returns the function to compute the error derivative of the output layer
  of a neural network from the target value and actual output for a given sample."
  [^Net nn]
  (case [(actfn-kind nn) (-> nn :training :params :cost-fn)]
    [:softmax :cross-entropy]  (fn [ts ys] (m/- ys ts))
    [:sigmoid :cross-entropy]  (fn [ts ys] (m/- ys ts))
    [:sigmoid :sum-of-squares] (fn [ts ys] (m/mult (m/mult ys (m/- 1.0 ys))
                                                   (m/- ys ts)))))

(defn error-derivatives-wrt-logit
  "Compute the error derivatives wrt the logit by backpropagating the error
  on the output from the last layer down to the first layer."
  [^Net nn activities targets]
  (let [dEdz-out ((output-layer-error-deriv nn) targets (last activities))
        ws (:weights nn)]
    (loop [i (- (count ws) 2), ws ws, as (butlast activities), dEdzs [dEdz-out]]
      (if (>= i 0)
        (let [dydz ((deriv-actfn (actfn-kind nn i)) (last as))
              dEdy (u/without-bias (m/* (first dEdzs) (last ws)))
              dEdz (m/mult dydz dEdy)]
          (recur (dec i) (butlast ws) (butlast as) (cons dEdz dEdzs)))
        (vec dEdzs)))))

(defn error-derivatives-wrt-weights
  "Compute the error derivatives wrt each weight by multiplying the 
  backpropagated error derivatives by each unit's inputs."
  [dEdzs inputs]
  (vec (pmap (fn [dEdz x] (m/* (m/t dEdz) (u/with-bias x))) dEdzs inputs)))

(defn error-derivatives
  "Compute the error derivatives wrt each weight by backpropagating the error
  on the output and multiplying each error derivative by the corresponding 
  unit's inputs."
  [^Net nn ^DataSet dset]
  (let [[x y] [(:x dset) (:y dset)]
        as    (net-activities nn x)
        dEdzs (error-derivatives-wrt-logit nn as y)
        dEdws (error-derivatives-wrt-weights dEdzs (cons x (butlast as)))]
    dEdws))

; - Weight update coefficients (learning rate / rprop step size)

(defn adapt-coef
  "Returns a function to update a single element of a matrix containing one 
  update coefficient per weight.  This 'coefficient' can be the learning rate 
  gain (for adaptive learning rate) or the step size in RProp.
  We pass an update function and the min and max bound for the coefficient."
  [updfn minc maxc dw pdw]
  {:pre [(= (m/size dw) (m/size pdw))]}
  (fn [i j c]
    (let [d (m/mget dw i j)
          p (m/mget pdw i j)
          newc (updfn d p c)]
      (if (> newc maxc) maxc
        (if (< newc minc) minc newc)))))

(defn adapt-weight-update-coef
  "Compute the weight update coefficient based on previous and current deltaw.
  This 'coefficient' can be the learning rate gain (for adaptive learning rate)
  or the step size in RProp.
  We pass an update function and the min and max bound for the coefficient."
  [updfn minc maxc prev-c deltaw prev-dw]
  {:pre [(or (nil? prev-c)
             (= (map m/size prev-c) (map m/size deltaw) (map m/size prev-dw)))]}
  (if (nil? prev-c)
    (vec (pmap (fn [dw] (apply m/ones (m/size dw))) deltaw))
    (vec (pmap (fn [pc dw pdw]
                 (m/map-indexed (adapt-coef updfn minc maxc dw pdw) pc))
               prev-c deltaw prev-dw))))

; - Learning rate

(defn adapt-learning-rate-gain
  "Compute the adaptive learning rate gain based on previous and current deltaw."
  [ming maxg prev-g deltaw prev-dw]
  (let [updfn (fn [d p g] (if (> (* d p) 0) (+ g 0.05) (* g 0.95)))]
    (adapt-weight-update-coef updfn ming maxg prev-g deltaw prev-dw)))

(defn adapt-rprop-step-size
  "Compute the new RProp step size based on previous and current deltaw."
  [mins maxs prev-s deltaw prev-dw]
  (let [updfn (fn [d p g] (if (> (* d p) 0) (* g 1.2) (* g 0.5)))]
    (adapt-weight-update-coef updfn mins maxs prev-s deltaw prev-dw)))

(defn apply-learning-rate
  "Apply the learning rate to deltaw.
  Returns the (possibly updated) neural net and the new deltaw."
  [^Net nn deltaw]
  {:pre [(-> nn :training :params :learning-rate :epsilon)]}
  (let [lr  (-> nn :training :params :learning-rate)
        eps (:epsilon lr)]
    (if (:adaptive lr)
      (let [[ming maxg] [(:ming lr) (:maxg lr)]
            prev-dw (-> nn :training :state :deltaw)
            prev-g  (-> nn :training :state :lr-gain)
            gain    (adapt-learning-rate-gain ming maxg prev-g deltaw prev-dw)
            nn      (assoc-in nn [:training :state :lr-gain] gain)
            dw      (vec (pmap #(m/mult (m/* eps %1) %2) gain deltaw))]
        [nn dw])
      [nn (vec (pmap (partial m/* eps) deltaw))])))

; - RProp step size

(defn adjust-step-size
  "Adjust RProp step size based on the gradient.
  Returns the (possibly updated) neural net and the new deltaw (step size)."
  [^Net nn deltaw]
  (let [rp (-> nn :training :params :rprop)
        [mins maxs] [(:mins rp) (:maxs rp)]
        prev-dw (-> nn :training :state :deltaw)
        prev-s  (-> nn :training :state :rp-step)
        step    (if prev-s
                  (adapt-rprop-step-size mins maxs prev-s deltaw prev-dw)
                  (vec (pmap (fn [dw] (m/* 0.01 (apply m/ones (m/size dw)))) deltaw)))
        nn      (assoc-in nn [:training :state :rp-step] step)
        dw      (vec (pmap #(m/mult %1 (m/signum %2)) step deltaw))]
    [nn dw]))

; - Root mean square gradient for RMSProp

(defn update-ms-gradient
  "Update the mean square gradient to be used in rmsprop."
  [prev-msg grad alpha]
  (if prev-msg
    (vec (pmap (fn [p g] (m/+ (m/* alpha p)
                              (m/* (- 1 alpha) (m/mult g g)))) prev-msg grad))
    (vec (pmap (fn [g] (m/mult g g)) grad))))

(defn apply-rms-gradient
  "Update mean square gradient, and apply it to compute deltaw."
  [^Net nn grad]
  (let [alpha    (-> nn :training :params :rmsprop :alpha)
        prev-msg (-> nn :training :state :ms-grad)
        ms-grad  (update-ms-gradient prev-msg grad alpha)
        nn       (assoc-in nn [:training :state :ms-grad] ms-grad)
        dw       (vec (pmap (fn [g m] (u/safe-div g (m/sqrt m))) grad ms-grad))]
    [nn dw]))

; - Momentum

(defn momentum-factor
  "Compute the momentum factor alpha, based on training parameters.
  Returns the (possibly updated) neural net and alpha (or nil, if not defined)."
  [^Net nn]
  {:pre [(or (nil? (-> nn :training :params :momentum))
             (-> nn :training :params :momentum :alpha))
         (or (nil? (-> nn :training :params :momentum :alpha-start))
             (-> nn :training :params :momentum :alpha-step))]}
  (if-let [momentum (-> nn :training :params :momentum)]
    (let [alpha    (if-let [prev-alpha (-> nn :training :state :alpha)]
                     (min (+ prev-alpha (:alpha-step momentum)) (:alpha momentum))
                     (or (:alpha-start momentum) (:alpha momentum)))
          nn       (if (:alpha-start momentum)
                     (assoc-in nn [:training :state :alpha] alpha)
                     nn)]
      [nn alpha])
    [nn nil]))

(defn apply-momentum
  "Apply the momentum to deltaw (if applicable).
  Returns the (possibly updated) neural net and the new deltaw."
  [^Net nn deltaw]
  (let [[nn alpha] (momentum-factor nn)
        prev-dw    (-> nn :training :state :deltaw)
        dw         (if (and alpha prev-dw)
                     (vec (pmap #(m/+ (m/* alpha %1) %2) prev-dw deltaw))
                     deltaw)]
    [nn dw]))

(defn apply-nesterov-momentum
  "Update the network weights by applying Nesterov's momentum.
  Returns the updated neural net and deltaw."
  [^Net nn]
  (let [[nn alpha] (momentum-factor nn)
        prev-dw    (-> nn :training :state :deltaw)
        dw         (if prev-dw
                     (vec (pmap (partial m/* alpha) prev-dw))
                     (vec (pmap #(apply m/zeros (m/size %)) (:weights nn))))
        nn         (assoc nn :weights (update-weights (:weights nn) dw))]
    [nn dw]))

; Delta weights for the different methods

(defmethod deltaw
  :backprop
  [^Net nn ^DataSet dset]
  (let [m       (first (m/size (:x dset)))
        dEdws   (error-derivatives nn dset)
        dw      (vec (pmap (partial m/* (/ -1.0 m)) dEdws))
        [nn dw] (apply-learning-rate nn dw)
        [nn dw] (apply-momentum nn dw)
        nn      (assoc-in nn [:training :state :deltaw] dw)]
    [nn dw]))

(defmethod deltaw
  :nesterov
  [^Net nn ^DataSet dset]
  (let [[nn prev-dw] (apply-nesterov-momentum nn)
        dEdws        (error-derivatives nn dset)
        m            (first (m/size (:x dset)))
        dw           (vec (pmap (partial m/* (/ -1.0 m)) dEdws))
        [nn dw]      (apply-learning-rate nn dw)
        acc-dw       (vec (pmap m/+ prev-dw dw))
        nn           (assoc-in nn [:training :state :deltaw] acc-dw)]
    [nn dw]))

(defmethod deltaw
  :rprop
  [^Net nn ^DataSet dset]
  (let [dEdws   (error-derivatives nn dset)
        dw      (vec (pmap (partial m/* -1.0) dEdws))
        [nn dw] (adjust-step-size nn dw)
        [nn dw] (apply-momentum nn dw)
        nn      (assoc-in nn [:training :state :deltaw] dw)]
    [nn dw]))

(defmethod deltaw
  :rmsprop
  [^Net nn ^DataSet dset]
  (let [dEdws   (error-derivatives nn dset)
        dw      (vec (pmap (partial m/* -1.0) dEdws))
        [nn dw] (apply-rms-gradient nn dw)
        [nn dw] (apply-learning-rate nn dw)
        [nn dw] (apply-momentum nn dw)
        nn      (assoc-in nn [:training :state :deltaw] dw)]
    [nn dw]))

(defn train-batch
  "Train the neural network with the provided batch dataset.
  Returns the updated neural network."
  [^Net nn ^DataSet dset]
  (let [[nn dw] (deltaw nn dset)]
    (assoc nn :weights (update-weights (:weights nn) dw))))

; Cost functions

(def ce-tiny 1e-30)

(defn misclassification
  "Misclassification cost function (for a single sample).
  Returns 1 if the sample is misclassified, 0 otherwise."
  [outputs targets]
  (if (= outputs targets) 0 1))

(defn sum-of-squares
  "Sum of square cost function (for a single sample)."
  [outputs targets]
  (reduce + (map (fn [o t] (Math/pow (- o t) 2)) outputs targets)))

(defn cross-entropy-binary
  "Binary cross-entropy cost function (for a single sample).
  Note it only works for binary target."  
  [outputs targets]
  (reduce + (map (fn [o t]
                   (- (if (= 1.0 (double t))
                        (Math/log (+ o ce-tiny))
                        (Math/log (- (+ 1 ce-tiny) o)))))
                 outputs targets)))

(defn cross-entropy-multivariate
  "Multivariate cross-entropy cost function (for a single sample).
  Note it only works for binary target."  
  [outputs targets]
  (reduce + (map (fn [o t]
                   (- (if (= 1.0 (double t)) (Math/log (+ o ce-tiny)) 0)))
                 outputs targets)))

(defn costfn-kind
  [^Net nn]
  (let [errorkind (-> nn :training :params :cost-fn)]
    (if (= errorkind :cross-entropy)
      (if (= (actfn-kind nn) :softmax)
        :cross-entropy-multivariate
        :cross-entropy-binary)
      errorkind)))

(defmulti cost-fn
  "Returns the cost function for a network or expressed as a keyword"
  (fn [arg] (type arg)))

(defmethod cost-fn
  clojure.lang.Keyword
  [errorkind]
  (resolve (symbol "synaptic.core" (name errorkind))))

(defmethod cost-fn
  Net
  [^Net nn]
  (cost-fn (costfn-kind nn)))

; L-BFGS learning

(defn weights-to-double-array
  "Returns a double-array with all weights of the network."
  [w]
  (double-array (apply concat (map flatten w))))

(defn double-array-to-weights
  "Recreate network weights from a double-array, based on the layer information."
  [d layers]
  (vec (loop [i 0, ds (vec d), ws []]
         (if (empty? ds)
           ws
           (let [r (nth layers (inc i))
                 c (inc (nth layers i))
                 n (* r c)
                 w (m/matrix (partition c (take n ds)))]
             (recur (inc i) (drop n ds) (conj ws w)))))))

(defn error+derivatives
  "Compute both the error and the error derivatives for a given dataset."
  [^Net nn ^DataSet dset]
  (let [[x y] [(:x dset) (:y dset)]
        as    (net-activities nn x)
        out   (last as)
        dEdzs (error-derivatives-wrt-logit nn as y)
        dEdws (error-derivatives-wrt-weights dEdzs (cons x (butlast as)))
        E     (reduce + (map (cost-fn nn) (m/dense out) (m/dense y)))]
    [(double E) (weights-to-double-array dEdws)]))

; Training error

(defn training-error-dataset
  "Compute a kind of training error of the network for a given dataset."
  [errorkind ^Net nn ^DataSet dset]
  (let [[x y] [(:x dset) (:y dset)]
        as    (last (net-activities nn x))
        out   (if (= :misclassification errorkind)
                (m/map (fn [^double a] (Math/round a)) as) as)
        cost  (cost-fn errorkind)]
    (reduce + (map cost (m/dense out) (m/dense y)))))

(defn training-error-datasets
  "Compute a kind of training error of the network for multiple datasets."
  [errorkind ^Net nn dsets]
  (let [ntrerr  (reduce + (pmap (partial training-error-dataset errorkind nn) dsets))
        m       (reduce + (map #(first (m/size (:x %))) dsets))]
    (/ (double ntrerr) m)))

(defn training-error
  "Compute the training error of a network on a given training set.
  Kind of errors are: :misclassification, :sum-of-square, :cross-entropy."
  [errorkind ^Net nn ^TrainingSet trset]
  (let [trerr (training-error-datasets errorkind nn (:batches trset))]
    (if-let [valds (:val-set trset)]
      (let [valerr (training-error-datasets errorkind nn [valds])]
        [trerr valerr])
      [trerr])))

; Training statistics

(defn error-kinds
  "Returns a vector of kinds of error to be computed as part of training stats.
  If not specified in training params, it defaults to the network cost function."
  [^Net nn]
  (or (-> nn :training :params :stats :errorkinds) [(costfn-kind nn)]))

(defn training-stats
  "Create training statistics for a neural network.  Stats will be updated
  as the training progresses."
  [^Net nn]
  (let [empty-errors (zipmap (error-kinds nn) (repeat []))]
    (Stats. 0 empty-errors empty-errors)))

(defn init-stats
  "Initialize training statistics."
  [^Net nn]
  (if (-> nn :training :stats)
    nn
    (assoc-in nn [:training :stats] (training-stats nn))))

(defn update-stats-errors
  "Update training and validation error estimates of a given kind 
  in the network training statistics."
  [errorkind ^Net nn ^TrainingSet trset]
  (let [[trerr valerr] (training-error errorkind nn trset)
        nn (update-in nn [:training :stats :tr-err errorkind] conj trerr)]
    (if valerr
      (update-in nn [:training :stats :val-err errorkind] conj valerr)
      nn)))

(defn update-stats
  "Update training statistics of the neural network, using a training set."
  [^Net nn ^TrainingSet trset]
  (let [nn (update-in nn [:training :stats :epochs] inc)]
    (loop [nn nn, errorkinds (error-kinds nn)]
      (if-let [errorkind (first errorkinds)]
        (recur (update-stats-errors errorkind nn trset) (rest errorkinds))
        nn))))

; Training on the whole training set

(defmulti train
  "Train the neural network (atom) on a given training set for a given number 
  of epochs (to be improved).
  Returns a future that will complete when the training ends.
  The neural network weights and training state and stats will be updated 
  as the training progresses."
  (fn [net _ _] (-> @net :training :algo)))

(defmethod train
  :lbfgs
  [net ^TrainingSet trset nepochs]
  (future
    (swap! net assoc-in [:training :state :state] :training)
    (swap! net init-stats)
    (let [l  (-> @net :arch :layers)
          b  (merge-batches (:batches trset))
          w0 (weights-to-double-array (:weights @net))
          wt (b/lbfgs (fn [^doubles w]
                        (swap! net assoc :weights (double-array-to-weights w l))
                        (swap! net update-stats trset)
                        (error+derivatives @net b))
                      w0 {:eps 0.1 :maxit nepochs})]
      (swap! net assoc :weights (double-array-to-weights wt l))
      (swap! net update-stats trset))
    (swap! net assoc-in [:training :state :state] nil)
    net))

(defn continue-training?
  "Returns true if the training should continue, that is, if the training
  has not been stopped and the max iteration is not reached yet."
  [net maxep]
  (and (not= :stopping (-> @net :training :state :state))
       (< (-> @net :training :stats :epochs) maxep)))

(defmethod train
  :default
  [net ^TrainingSet trset nepochs]
  (future
    (swap! net assoc-in [:training :state :state] :training)
    (swap! net init-stats)
    (let [maxep (+ nepochs (-> @net :training :stats :epochs))
          all-batches (:batches trset)]
      (loop [batches all-batches]
        (when (continue-training? net maxep)
          (swap! net train-batch (first batches))
          (if (next batches)
            (recur (rest batches))
            (do
              (swap! net update-stats trset)
              (recur all-batches))))))
    (swap! net assoc-in [:training :state :state] nil)
    net))

(defmulti stop-training
  "Stop the training of a neural net.  This operation is not supported when
  using LBFGS training algorithm."
  (fn [net] (-> @net :training :algo)))

(defmethod stop-training
  :lbfgs
  [net]
  (if (= :training (-> @net :training :state :state))
    (throw (IllegalStateException. "Training cannot be stopped with LBFGS"))))

(defmethod stop-training
  :default
  [net]
  (if (= :training (-> @net :training :state :state))
    (swap! net assoc-in [:training :state :state] :stopping)))


