(ns
  ^{:doc "synaptic - convolution operations"
    :author "Antoine Choppin"}
  synaptic.conv
  (:require [clatrix.core :as m])
  (:gen-class))


(defn cnv-gather!
  "Compute a single element of the convolution matrix, for all samples.
  Each sample (row) of 'a' is convoluted with the weight matrix 'b'.
  This function assumes the existence of the result matrix 'c' and fills up 
  one column (corresponding to 1 element of the convolution matrix for all 
  samples).
  If 'a' has multiple features/channels (i.e. if 'ka' > 1), the convolution 
  is computed for all channels and summed up.
   'a' = (input)  sample matrix       n  x (ka * wa * ha)
   'b' = (input)  weight matrix       kb x (ka * wb * hb)
   'c' = (output) convolution matrix  n  x (kb * wc * hc)
    n  = number of samples
    k  = current output channel (in 1..kb)
   i,j = coordinates of the element to compute in 'c'
   ijc = 1D index of the (k,i,j)th element in 'c' (output column index)"
  [a b c ka wa wha wb hb i j k ijc]
  (let [n   (int (first (m/size a)))
        whb (int (* wb hb))
        bk  (double-array (first (m/rows b [k])))
        r   (double-array (repeat n (aget bk 0)))]
    (dotimes [kk ka]
      (let [kkwha (int (* kk wha))
            kkwhb (int (* kk whb))]
        (dotimes [ii hb]
          (let [iiwa (int (* (+ i ii) wa))
                iiwb (int (* ii wb))]
          (dotimes [jj wb]
            (let [t   (int (+ kkwhb iiwb jj 1))
                  ija (int (+ kkwha iiwa j jj))]
              (dotimes [p n]
                (aset r p (+ (aget r p)
                             (* (m/mget a p ija)
                                (aget bk t)))))))))))
    (dotimes [p n] (m/mset! c p ijc (aget r p)))))

(defn convolute
  "Compute the convolution of each sample in 'a' with the weight matrix 'b'.
  Each sample is a row of 'a', and each convoluted sample is a row of the
  output matrix 'c'.
  If 'a' has multiple features/channels (i.e. if 'ka' > 1), the convolution 
  is computed for all channels and summed up.
   'a' = (input)  sample matrix       n  x (ka * wa * ha)
   'b' = (input)  weight matrix       kb x (ka * wb * hb)
   'c' = (output) convolution matrix  n  x (kb * wc * hc)"
  [a b ka wa ha kb wb hb]
  (let [wha (int (* wa ha))
        wc  (int (- wa (dec wb)))
        hc  (int (- ha (dec hb)))
        whc (int (* wc hc))
        c   (m/zeros (first (m/size a)) (* kb whc))]
    (doall (pmap (fn [k]
                   (dotimes [i hc]
                     (dotimes [j wc]
                       (let [ijc (+ (* k whc) (* i wc) j)]
                         (cnv-gather! a b c ka wa wha wb hb i j k ijc)))))
                 (range kb)))
    c))

(defn max-gather!
  "Compute for each sample a single cell of the max pooled matrix 'd' AND 
  store the index of the maximum in 'is'.  So the result is 2 columns, 
  1 column in 'd' (the maximum values of a max pooling pad for each sample)
  and 1 column in 'is' (the index of the max value for each sample).
    'c'  = (input)  convolution matrix  n  x (kb * wc * hc)
    'd'  = (output) max pooled matrix   n  x (kb * wout * hout)
    'is' = (output) indices matrix      n  x (kb * wout * hout)
     n   = number of samples
     k   = current output channel (in 1..kb)
   wpool,hpool = size of the pooling pad
   iout,jout = coordinates of the element to compute in 'd'
   ijout = 1D index of the (k,i,j)th element in 'd' (output column index)"
  [c d is n k wpool hpool wc whc iout jout ijout]
  (let [mv    (double-array (repeat n Double/NEGATIVE_INFINITY))
        mi    (int-array    (repeat n 0))
        ijout (int ijout)]
    (dotimes [ipool hpool]
      (let [ic (int (+ (* k whc) (* (+ (* iout hpool) ipool) wc)))]
        (dotimes [jpool wpool]
          (let [ijc (int (+ ic (* jout wpool) jpool))]
            (dotimes [p n]
              (let [v (m/mget c p ijc)]
                (when (> v (aget mv p))
                  (aset mv p v)
                  (aset mi p ijc))))))))
    (dotimes [p n]
      (m/mset! d  p ijout (aget mv p))
      (m/mset! is p ijout (aget mi p)))))

(defn pooling-activities-and-indices
  "Compute max pooling matrix AND indices of the max value in the original
  matrix, for each sample of 'c'.  Returns a map {:a activities :i indices}.
    'c'  = (input)  convolution matrix  n  x (kc * wc * hc)
    'd'  = (output) max pooled matrix   n  x (kc * wout * hout)
    'is' = (output) indices matrix      n  x (kc * wout * hout)
   wpool,hpool = size of the pooling pad
   wout,hout   = size of the output (pooled) matrix
  Currently, only :max pool-kind is supported."
  [c kc pool-kind wpool hpool wout hout]
  {:pre [(= pool-kind :max)]}
  (let [n       (int (first (m/size c)))
        [wc hc] [(int (* wout wpool)) (int (* hout hpool))]
        whc     (int (* wc hc))
        whout   (int (* wout hout))
        kwhout  (int (* kc whout))
        is      (m/zeros n kwhout)
        d       (m/zeros n kwhout)]
    (doall (pmap (fn [k]
                   (dotimes [iout hout]
                     (let [ii (int (+ (* k whout) (* iout wout)))]
                       (dotimes [jout wout]
                         (let [ijout (int (+ ii jout))]
                           (max-gather! c d is n k wpool hpool
                                        wc whc iout jout ijout))))))
                 (range kc)))
    {:a d :i is}))

(defn sum-gather
  "Compute the sum of contributions to a given weight, that is, the sum of 
  the products of each input element (in 'a') with the corresponding element 
  in 'dc'.  The contributions to a given weight are summed over all input 
  samples.
   'a'  = (input)  sample matrix       n  x (ka * wa * ha)
   'c'  = (input)  error derivatives   n  x (kc * wc * hc)
    k   = current output channel (in 1..kb)
  bi,bj = coordinates of the element to compute in db"
  [a dc wa wha wc hc k bi bj]
  (let [n    (int (first (m/size a)))
        r    (double-array [0.0])
        bi   (int bi)
        bj   (int bj)
        kwha (int (* k wha))]
    (dotimes [ii hc]
      (let [iiwc (int (* ii wc))
            iiwa (int (* (+ ii bi) wa))]
        (dotimes [jj wc]
          (let [t   (int (+ iiwc jj))
                ija (int (+ kwha iiwa (+ jj bj)))]
            (dotimes [p n]
              (aset r 0 (+ (aget r 0) (* (m/mget a p ija) (m/mget dc p t)))))))))
    (aget r 0)))

(defn sum-contributions
  "Compute the sum of contributions to the weights.
  The contributions of all input samples to a given weight are summed up.
   'a'  = (input)  sample matrix       n  x (ka * wa * ha)
   'c'  = (input)  error derivatives   n  x (kc * wc * hc)
  bi,bj = coordinates of the element to compute in db"
  [a dc ka wa ha wc hc]
  (let [wha   (int (* wa ha))
        wb    (int (inc (- wa wc)))
        hb    (int (inc (- ha hc)))
        whb   (int (* wb hb))
        whc   (int (* wc hc))
        kwhb1 (int (inc (* ka whb)))
        kb    (int (/ (second (m/size dc)) whc))
        n     (first (m/size a))
        db    (m/zeros kb kwhb1)
        onesc (m/ones n whc)]
    (doall (pmap
      (fn [i]
        (let [dck (apply m/hstack (m/cols dc (range (* i whc) (* (inc i) whc))))]
          (m/mset! db i 0 (sum-gather onesc dck wc 0 wc hc 0 0 0))
          (dotimes [k ka]
            (let [jj (inc (* k whb))]
              (dotimes [bi hb]
                (dotimes [bj wb]
                  (m/mset! db i (+ jj (* bi wb) bj)
                           (sum-gather a dck wa wha wc hc k bi bj))))))))
      (range kb)))
    db))

(defmacro dofromto
  [[ii from to] & code]
  `(let [start# ~from
         end#   ~to]
     (loop [~ii start#]
       (when (< ~ii end#)
         ~@code
         (recur (inc ~ii))))))

(defn fromto
  [i hb ha]
  [(int (max 0 (- hb i 1))) (int (min hb (- ha i)))])

(defn rev-cnv-gather!
  "Compute a single element of the 'reverse convolution' matrix, 
  for all samples.  Each sample (row) of 'dc' is convoluted with the weight 
  matrix 'b' flipped about both axes.  The convolution spans over elements 
  outside of the bounds of 'dc', which are considered to be 0.0 (this is like 
  if we padded 'dc' on 4 sides before doing the convolution).
  This function assumes the existence of the result matrix 'da' and fills up 
  one column (corresponding to 1 element of the convolution matrix for all 
  samples).
  If 'dc' has multiple features/channels (i.e. if 'kc' > 1), the convolution 
  is computed for all channels and summed up.
  'dc' = (input)  error derivatives matrix      n  x (kc * wc * hc)
   'b' = (input)  weight matrix                 kb x (kc * wb * hb)
  'da' = (output) 'reverse convolution' matrix  n  x (ka * wa * ha)
    n  = number of samples
    k  = current output channel (in 1..kb)
   i,j = coordinates of the element to compute in 'da'
   ija = 1D index of the (k,i,j)th element in 'da' (output column index)"
  [da b dc wa ha kc wc hc wb hb i j k ija]
  (let [n     (int (first (m/size da)))
        whb   (int (* wb hb))
        whc   (int (* wc hc))
        bcols (range (inc (* k whb)) (inc (* (inc k) whb)))
        bk    (double-array
                (apply m/hstack (m/rows
                  (apply m/hstack (reverse (m/cols b bcols))))))
        r     (double-array (repeat n 0))
        ic    (int (- i (dec hb)))
        jc    (int (- j (dec wb)))]
    (dotimes [kk kc]
      (let [kkwhb   (int (* kk whb))
            kkwhc   (int (* kk whc))
            [i1 i2] (fromto i hb ha)]
        (dofromto [ii i1 i2]
          (let [iiwb    (int (* ii wb))
                iiwc    (int (* (+ ic ii) wc))
                [j1 j2] (fromto j wb wa)]
            (dofromto [jj j1 j2]
              (let [t   (int (+ kkwhb iiwb jj))
                    ijc (int (+ kkwhc iiwc jc jj))]
                (dotimes [p n]
                  (aset r p (+ (aget r p)
                               (* (m/mget dc p ijc)
                                  (aget bk t)))))))))))
    (dotimes [p n] (m/mset! da p ija (aget r p)))))

(defn backprop-dEdz
  "Compute the 'reverse convolution' of each sample of 'dc' with the weight 
  matrix 'b' flipped about both axes.  The convolution spans over elements 
  outside of the bounds of 'dc', which are considered to be 0.0 (this is like 
  if we padded 'dc' on 4 sides before doing the convolution).
  If 'dc' has multiple features/channels (i.e. if 'kc' > 1), the convolution 
  is computed for all channels and summed up.
  'dc' = (input)  error derivatives matrix      n  x (kc * wc * hc)
   'b' = (input)  weight matrix                 kb x (kc * wb * hb)
  'da' = (output) 'reverse convolution' matrix  n  x (ka * wa * ha)"
  [dc b ka wa ha kc wc hc]
  (let [wb  (int (inc (- wa wc)))
        hb  (int (inc (- ha hc)))
        wha (int (* wa ha))
        da  (m/zeros (first (m/size dc)) (* ka wha))]
    (doall (pmap (fn [k]
                   (dotimes [i ha]
                     (dotimes [j wa]
                       (let [ija (+ (* k wha) (* i wa) j)]
                         (rev-cnv-gather! da b dc wa ha kc wc hc wb hb i j k ija)))))
                 (range ka)))
    da))

