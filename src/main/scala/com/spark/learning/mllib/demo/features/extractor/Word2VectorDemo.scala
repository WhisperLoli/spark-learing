package com.spark.learning.mllib.demo.features.extractor

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.feature.Word2Vec

/**
  *  Word2Vector将词转换成分布式向量。分布式表示的主要优势是相似的词在向量空间距离较近
  */
object Word2VectorDemo extends App {
  val documentDF = spark.createDataFrame(Seq(
    "Hi I heard about Spark".split(" "),
    "I wish Java could use case classes".split(" "),
    "Logistic regression models are neat".split(" ")
  ).map(Tuple1.apply)).toDF("text")

  // Learn a mapping from words to Vectors.
  val word2Vec = new Word2Vec()
    .setInputCol("text")
    .setOutputCol("result")
    .setVectorSize(3)
    .setMinCount(0)
  val model = word2Vec.fit(documentDF)

  val result = model.transform(documentDF)
  result.show(false)
}
