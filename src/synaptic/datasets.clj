(ns
  ^{:doc "synaptic - data sets"
    :author "Antoine Choppin"}
  synaptic.datasets
  (:require [clatrix.core :as m]
            [synaptic.util :as u])
  (:gen-class))


(defrecord DataSet [x y])
(defrecord TrainingSet [header batches valid])

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
  (let [lbsmp (u/load-data "labeledsamples" sname)
        fsize (-> lbsmp :header :fieldsize)]
    (assoc lbsmp :samples
                 (mapv (partial decode-sample fsize) (:samples lbsmp)))))

(defn save-training-set
  "Save training set to disk.
  The header is saved separately for quicker access."
  [^TrainingSet trset]
  (let [header  (:header trset)
        tsname  (header :name)]
    (u/save-data "trainingset-header" header tsname)
    (u/save-data "trainingset"        trset  tsname)))

(defn load-training-set-header
  "Load a training set header from disk."
  [tsname]
  (u/load-data "trainingset-header" tsname))

(defn load-training-set
  "Load a training set from disk."
  [tsname]
  (u/load-data "trainingset" tsname))

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
  (if (and batchsize (< 0 batchsize (count (first vectors))))
    (mapv (comp vec (partial map vec) (partial partition-all batchsize)) vectors)
    (mapv vector vectors)))

(defn dataset
  "Create a dataset with x and (optionally) y."
  ([x y]
   {:pre [(= (count x) (count y))]}
   (DataSet. (m/matrix x) (m/matrix y)))
  ([x]
   (DataSet. (m/matrix x) nil)))

(defn merge-batches
  "Merge batches into a single dataset."
  [batches]
  (if (> (count batches) 1)
    (DataSet. (apply m/vstack (map :x batches)) (apply m/vstack (map :y batches)))
    (first batches)))

(defn count-labels
  "Create a map with number of occurrence of each label."
  [labeltranslator encodedlabels]
  (let [translate-keys #(zipmap (mapv labeltranslator (keys %))
                                (vals %))]
    (translate-keys (frequencies encodedlabels))))

(defn training-set
  "Create a training set from samples and associated labels.
  The training set consists of one or more batches and optionally a validation set.
  It also has a map that will allow converting y's back to the original labels.
  
  Options:
    :name           - a name for the training set
    :type           - the type of training data (e.g. :binary-image, :grayscale-image ...)
    :continuous true - set this flag to use continuous labels (auto-scaled to between 0 and 1)
    :fieldsize      - [width height] of each sample data (for images)
    :nvalid         - size of the validation set (default is 0, i.e. no validation set)
    :batch          - size of a mini-batch (default is the number of samples, after
                      having set apart the validation set)
    :online true    - set this flag for online training (same as batch size = 1)
    :rand false     - unset this flag to keep original ordering (by default, samples
                      will be shuffled before partitioning)."
  [samples labels & [options]]
  {:pre [(= (count samples) (count labels))]}
  (let [batchsize  (if (:online options) 1 (:batch options))
        trainsize  (if (:nvalid options) (- (count samples) (:nvalid options)))
        randomize  (if (nil? (:rand options)) true (:rand options))
        [reglb lbtranslator]  (if (:continuous options) (u/tocontinuous labels) (u/tobinary labels))
        [smp lb]   (if randomize (shuffle-vecs samples reglb) [samples reglb])
        [trainsmp validsmp] (if trainsize (split-at trainsize smp) [smp nil])
        [trainlb  validlb]  (if trainsize (split-at trainsize lb) [lb nil])
        [batchsmp batchlb]  (partition-vecs batchsize trainsmp trainlb)
        trainsets  (mapv dataset batchsmp batchlb)
        validset   (if trainsize (dataset validsmp validlb))
        timestamp  (System/currentTimeMillis)
        header     {:name (or (:name options) timestamp)
                    :timestamp timestamp
                    :type (:type options)
                    :fieldsize (or (:fieldsize options)
                                   (u/divisors (count (first samples))))
                    :batches (mapv (partial count-labels lbtranslator) batchlb)
                    :valid (count-labels lbtranslator validlb)
                    :labeltranslator lbtranslator}]
    (TrainingSet. header trainsets validset)))

(defn test-set
  "Create a test set from samples.
  The test set consists of a single batch.
  
  Options:
    :name        - a name for the test set
    :type        - the type of test data (e.g. :binary-image, :grayscale-image ...)
    :fieldsize   - [width height] of each sample data (for images)
    :rand true   - set this flag to shuffle samples."
  [samples & [options]]
  (let [randomize (if (nil? (:rand options)) false (:rand options))
        testset   (dataset (if randomize (shuffle-vecs samples) samples))
        timestamp (System/currentTimeMillis)
        header    {:name (or (:name options) timestamp)
                   :timestamp timestamp
                   :type (:type options)
                   :batches [{"?" (count samples)}]
                   :fieldsize (or (:fieldsize options)
                                  (u/divisors (count (first samples))))}]
    (TrainingSet. header [testset] nil)))

