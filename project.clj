(defproject synaptic "0.1.0"
  :description "Neural Networks in Clojure"
  :url "https://github.com/japonophile/synaptic"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.6.0"]
                 [clatrix/clatrix "0.3.0"]
                 [bozo/bozo "0.1.1"]]
  :aot :all
  :eastwood {:exclude-linters [:unused-ret-vals]})
