(ns
  ^{:doc "synaptic - training"
    :author "Antoine Choppin"}
  synaptic.training
  (:require [clatrix.core :as m]
            [synaptic.net :as n]
            [synaptic.datasets :as d]
            [synaptic.util :as u]
            [bozo.core :as b])
  (:import  [synaptic.net Net]
            [synaptic.datasets DataSet TrainingSet]
            [org.jblas DoubleMatrix])
  (:gen-class))


(defrecord Stats [epochs tr-err val-err])
(defrecord Training [algo params state stats])

(defn training
  "Initialize network training for a given algorithm with optional training 
  parameters.  Currently supported algorithms are :perceptron, :backprop,
  :rprop, :rmsprop and :lbfgs.

  Training parameters:
  {
    :error-fn           :misclassification, :sum-of-squares, :cross-entropy
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
    :regularization {
        :lambda         regularization coefficient
        :kind           :l1 or :l2
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
          :perceptron {:error-fn :misclassification}
          :backprop   {:error-fn :cross-entropy
                       :learning-rate {:epsilon 0.01}}
          :rprop      {:error-fn :cross-entropy
                       :rprop {:mins 0.000001 :maxs 50.0}}
          :rmsprop    {:error-fn :cross-entropy
                       :rmsprop {:alpha 0.9}
                       :learning-rate {:epsilon 0.01}}
          :lbfgs      {:error-fn :cross-entropy})]
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
        y     (:a (last (n/net-activities nn x)))
        err   (perceptron-error y t)]
    [nn [(m/* (m/t err) (u/with-bias x))]]))

; Backpropagation learning

; - Error derivatives

(defn deriv-actfn
  "Returns the derivative of an activation function (expressed as a keyword)."
  [actkind]
  (case actkind
    :softmax            (fn [ys] (m/mult ys (m/- 1.0 ys)))
    :sigmoid            (fn [ys] (m/mult ys (m/- 1.0 ys)))
    :hyperbolic-tangent (fn [ys] (m/- 1.0 (m/mult ys ys)))))

(defn output-layer-error-deriv
  "Returns the function to compute the error derivative of the output layer
  of a neural network from the target value and actual output for a given sample."
  [^Net nn]
  (case [(n/actfn-kind nn) (-> nn :training :params :error-fn)]
    [:softmax :cross-entropy]  (fn [ts ys] (m/- ys ts))
    [:sigmoid :cross-entropy]  (fn [ts ys] (m/- ys ts))
    [:sigmoid :sum-of-squares] (fn [ts ys] (m/mult (m/mult ys (m/- 1.0 ys))
                                                   (m/- ys ts)))))

(defmulti prev-layer-error-deriv-wrt-logit
  "Compute the previous layer dEdz, based on dEdz-out, w (weights of the current
  layer) and y (activities for the *previous* layer)."
  (fn [layers l dEdz-out ay w] (:type (nth layers l))))

(defmulti layer-error-deriv-wrt-weights
  "Compute the layer dEdw, based on dEdz-out and x and y."
  (fn [layers l dEdz-out ax ay] (:type (nth layers l))))

(defn error-derivatives-wrt-logit
  "Compute the error derivatives wrt logit for all layers of the network, by 
  backpropagating it from the output layer down to the first layer."
  [layers dEdz-out ws ys]
  (let [last-layer-l (dec (count layers))]
    (loop [l last-layer-l, ws ws, ys (butlast ys), dEdzs [dEdz-out]]
      (if (> l 1)
        (let [dEdz (prev-layer-error-deriv-wrt-logit
                     layers l (last dEdzs) (last ys) (last ws))]
          (recur (dec l) (butlast ws) (butlast ys) (cons dEdz dEdzs)))
        (vec dEdzs)))))

(defn error-derivatives-wrt-weights
  "Based on error derivatives wrt logit, compute the error derivatives wrt weights 
  for all layers of the network."
  [layers dEdzs xs ys]
  {:pre [(= (dec (count layers)) (count dEdzs) (count xs) (count ys))]}
  (mapv (fn [l dEdz x y]
          (layer-error-deriv-wrt-weights layers l dEdz x y))
        (range 1 (count layers)) dEdzs xs ys))

(defn backpropagate-error-derivatives
  "Compute the error derivatives wrt weights for all layers of the network, 
  based on input, targets and pre-computed activities."
  [^Net nn inputs targets activities]
  (let [dEdz-out ((output-layer-error-deriv nn) targets (:a (last activities)))
        layers   (-> nn :arch :layers)
        ws       (:weights nn)
        [xs ys]  [(cons {:a inputs} (butlast activities)) activities]
        dEdzs    (error-derivatives-wrt-logit layers dEdz-out ws ys)
        dEdws    (error-derivatives-wrt-weights layers dEdzs xs ys)]
    dEdws))

(defn error-derivatives
  "Compute the error derivatives wrt each weight by backpropagating the error
  on the output and multiplying each error derivative by the corresponding 
  unit's inputs."
  [^Net nn ^DataSet dset]
  (let [[x y] [(:x dset) (:y dset)]
        as    (n/net-activities nn x)
        dEdws (backpropagate-error-derivatives nn x y as)]
    dEdws))

; Regularization

(defn norm-fn
  "Returns the norm function (to be used for regularization)."
  [normkind lambda]
  (case normkind
    :l1 (fn [w] (* 0.5 lambda (if (> 0.0 w) (- w) w)))
    :l2 (fn [w] (* 0.5 lambda w w))
    (fn [_] 0.0))) ; unknown

(defn regularization-penalty
  "Compute the regularization penalty for either L1 or L2 square norm
  (multiplied by the regularization coefficient).
  Note: the bias does not contribute to the penalty."
  [reg weights]
  (let [reg-fn (norm-fn (:kind reg) (:lambda reg))]
    (reduce + (pmap (fn [ws]
                      (reduce +
                        (map reg-fn (flatten (u/without-bias ws))))) weights))))

(defn deriv-norm-fn
  "Returns the derivative of a norm function, expressed as a keyword
  (to be used for regularization)."
  [normkind lambda]
  (case normkind
    :l1 (fn [w] (if (> 0.0 w) (- lambda) lambda))
    :l2 (fn [w] (* lambda w))
    (fn [_] 0.0))) ; unknown

(defn regularization-derivatives
  "Compute the regularization term of the cost derivatives for either L1 or
  L2 square norm (multiplied by the regularization coefficient).
  Note: the bias is not regularized (corresponding term is always 0.0)."
  [reg weights]
  (let [reg-fn (deriv-norm-fn (:kind reg) (:lambda reg))]
    (vec (pmap (fn [ws]
                 (m/map-indexed (fn [i j w]
                                  ; don't regularize the bias
                                  (if (> j 0) (reg-fn w) 0.0)) ws))
               weights))))

(defn cost-derivatives
  "Compute the derivatives of the cost function (including error and optionally
  a regularization penaly) wrt each weight.  If there is no regularization term,
  this is just error-derivatives."
  [^Net nn ^DataSet dset]
  (let [dEdws (error-derivatives nn dset)
        reg   (-> nn :training :parameters :regularization)]
    (if reg
      (vec (pmap (fn [dEdw r] (m/+ dEdw r))
                 dEdws (regularization-derivatives reg (:weights nn))))
      dEdws)))

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
        dCdws   (cost-derivatives nn dset)
        dw      (vec (pmap (partial m/* (/ -1.0 m)) dCdws))
        [nn dw] (apply-learning-rate nn dw)
        [nn dw] (apply-momentum nn dw)
        nn      (assoc-in nn [:training :state :deltaw] dw)]
    [nn dw]))

(defmethod deltaw
  :nesterov
  [^Net nn ^DataSet dset]
  (let [[nn prev-dw] (apply-nesterov-momentum nn)
        dCdws        (cost-derivatives nn dset)
        m            (first (m/size (:x dset)))
        dw           (vec (pmap (partial m/* (/ -1.0 m)) dCdws))
        [nn dw]      (apply-learning-rate nn dw)
        acc-dw       (vec (pmap m/+ prev-dw dw))
        nn           (assoc-in nn [:training :state :deltaw] acc-dw)]
    [nn dw]))

(defmethod deltaw
  :rprop
  [^Net nn ^DataSet dset]
  (let [dCdws   (cost-derivatives nn dset)
        dw      (vec (pmap (partial m/* -1.0) dCdws))
        [nn dw] (adjust-step-size nn dw)
        [nn dw] (apply-momentum nn dw)
        nn      (assoc-in nn [:training :state :deltaw] dw)]
    [nn dw]))

(defmethod deltaw
  :rmsprop
  [^Net nn ^DataSet dset]
  (let [dCdws   (cost-derivatives nn dset)
        dw      (vec (pmap (partial m/* -1.0) dCdws))
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

; Error functions

(def ce-tiny 1e-30)

(defn misclassification
  "Misclassification error function (for a single sample).
  Returns 1 if the sample is misclassified, 0 otherwise."
  [outputs targets]
  (if (= outputs targets) 0 1))

(defn sum-of-squares
  "Sum of square error function (for a single sample)."
  [outputs targets]
  (reduce + (map (fn [o t] (Math/pow (- o t) 2)) outputs targets)))

(defn cross-entropy-binary
  "Binary cross-entropy error function (for a single sample).
  Note it only works for binary target."  
  [outputs targets]
  (reduce + (map (fn [o t]
                   (- (if (= 1.0 (double t))
                        (Math/log (+ o ce-tiny))
                        (Math/log (- (+ 1 ce-tiny) o)))))
                 outputs targets)))

(defn cross-entropy-multivariate
  "Multivariate cross-entropy error function (for a single sample).
  Note it only works for binary target."  
  [outputs targets]
  (reduce + (map (fn [o t]
                   (- (if (= 1.0 (double t)) (Math/log (+ o ce-tiny)) 0)))
                 outputs targets)))

(defn specific-error-kind
  [^Net nn errorkind]
  (if (= errorkind :cross-entropy)
    (if (= (n/actfn-kind nn) :softmax)
      :cross-entropy-multivariate
      :cross-entropy-binary)
    errorkind))

(defn errorfn-kind
  [^Net nn]
  (let [errorkind (-> nn :training :params :error-fn)]
    (specific-error-kind nn errorkind)))

(defmulti error-fn
  "Returns the error function for a network or expressed as a keyword"
  (fn [arg] (type arg)))

(defmethod error-fn
  clojure.lang.Keyword
  [errorkind]
  (resolve (symbol "synaptic.training" (name errorkind))))

(defmethod error-fn
  Net
  [^Net nn]
  (error-fn (errorfn-kind nn)))

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
           (let [r (n/layer-units layers (inc i))
                 c (inc (n/layer-units layers i))
                 n (* r c)
                 w (m/matrix (partition c (take n ds)))]
             (recur (inc i) (drop n ds) (conj ws w)))))))

(defn error+derivatives
  "Compute both the error and the error derivatives for a given dataset."
  [^Net nn ^DataSet dset]
  (let [[x y] [(:x dset) (:y dset)]
        as    (n/net-activities nn x)
        dEdws (backpropagate-error-derivatives nn x y as)
        E     (reduce + (map (error-fn nn) (m/dense (:a (last as))) (m/dense y)))]
    [(double E) dEdws]))

(defn cost+derivatives
  "Compute both the cost and the cost derivatives for a given dataset."
  [^Net nn ^DataSet dset]
  (let [[E dEdws] (error+derivatives nn dset)
        weights   (:weights nn)
        reg       (-> nn :training :parameters :regularization)]
    (if reg
      [(+ E (regularization-penalty reg weights))
       (vec (pmap (fn [dEdw r] (m/+ dEdw r))
                  dEdws (regularization-derivatives reg weights)))]
      [E dEdws])))

; Training error

(defn training-error-dataset
  "Compute a kind of training error of the network for a given dataset."
  [errorkind ^Net nn ^DataSet dset]
  (let [[x y] [(:x dset) (:y dset)]
        as    (:a (last (n/net-activities nn x)))
        out   (if (= :misclassification errorkind)
                (m/map (fn [^double a] (Math/round a)) as) as)
        errfn (error-fn errorkind)]
    (reduce + (map errfn (m/dense out) (m/dense y)))))

(defn training-error-datasets
  "Compute a kind of training error of the network for multiple datasets."
  [errorkind ^Net nn dsets]
  (let [ntrerr  (reduce + (pmap (partial training-error-dataset errorkind nn) dsets))
        m       (reduce + (map #(first (m/size (:x %))) dsets))]
    (/ (double ntrerr) m)))

(defn net-error-dataset
  "Compute the network error for a given dataset."
  [^Net nn ^DataSet dset]
  (training-error-dataset (errorfn-kind nn) nn dset))

(defn training-error
  "Compute the training error of a network on a given training set.
  Kind of errors are: :misclassification, :sum-of-square, :cross-entropy."
  [errorkind ^Net nn ^TrainingSet trset]
  (let [trerr (training-error-datasets errorkind nn (:batches trset))]
    (if-let [valds (:valid trset)]
      (let [valerr (training-error-datasets errorkind nn [valds])]
        [trerr valerr])
      [trerr])))

; Training statistics

(defn error-kinds
  "Returns a vector of kinds of error to be computed as part of training stats.
  If not specified in training params, it defaults to the network error function."
  [^Net nn]
  (if-let [errorkinds (-> nn :training :params :stats :errorkinds)]
    (mapv (partial specific-error-kind nn) errorkinds)
    [(errorfn-kind nn)]))

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
    (let [l         (-> @net :arch :layers)
          b         (d/merge-batches (:batches trset))
          w0        (weights-to-double-array (:weights @net))
          [C dCdws] (cost+derivatives @net b)
          wt        (b/lbfgs (fn [^doubles w]
                               (swap! net assoc :weights (double-array-to-weights w l))
                               (swap! net update-stats trset)
                               [C (weights-to-double-array dCdws)])
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

