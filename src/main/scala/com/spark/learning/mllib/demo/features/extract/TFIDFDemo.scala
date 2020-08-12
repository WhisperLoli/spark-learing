package com.spark.learning.mllib.demo.features.extract

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}

object TFIDFDemo extends App {
  spark.sparkContext.setLogLevel("WARN")

  val sentenceData = spark.createDataFrame(Seq(
    (0, "I heard about Spark and I love Spark"),
    (0, "I wish Java could use case classes Java Java case"),
    (1, "Logistic regression models are neat")
  )).toDF("label", "sentence")

  // 分解器分解成词袋
  val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
  val wordsData = tokenizer.transform(sentenceData)
  wordsData.show()

  // 统计词频 TF: 词频，生成的rawFeatures中第二个数组hash后对应索引值，rawFeatures为稀疏向量
  // 2000为hash大小，太小容易出现不同字符串映射到同一个索引，出现冲突，再根据索引计算IDF
  // 索引值公式 hash(term)%numFeatures
  val hashingTF = new HashingTF().setNumFeatures(2000)
    .setInputCol("words").setOutputCol("rawFeatures")
  val featurizedData = hashingTF.transform(wordsData)
  // alternatively, CountVectorizer can also be used to get term frequency vectors
  featurizedData.show(false)

  // IDF: 逆文本频率指数
  // 如果某个词在所有的文章中少见，但是它在这篇文章中多次出现，那么它很可能就反映了这篇文章的特性，正是我们所需要的关键词
  val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
  val idfModel = idf.fit(featurizedData)

  val rescaledData = idfModel.transform(featurizedData)
  rescaledData.select("label", "features").show(false)

}
