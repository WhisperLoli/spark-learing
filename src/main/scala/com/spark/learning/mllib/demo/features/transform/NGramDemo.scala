package com.spark.learning.mllib.demo.features.transform

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.feature.NGram

/**
  * 将n个单词按顺序组成一组的所有组合情况，Tokenizer用于分割句子
  */
object NGramDemo extends App {
  val wordDataFrame = spark.createDataFrame(Seq(
    (0, Array("Hi", "I", "heard", "about", "Spark")),
    (1, Array("I", "wish", "Java", "could", "use", "case", "classes")),
    (2, Array("Logistic", "regression", "models", "are", "neat"))
  )).toDF("id", "words")

  val ngram = new NGram().setN(3).setInputCol("words").setOutputCol("ngrams")

  val ngramDataFrame = ngram.transform(wordDataFrame)
  ngramDataFrame.select("ngrams").show(false)
}
