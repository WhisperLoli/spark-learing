package com.spark.learning.mllib.demo.classification

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
/**
  * 朴素贝叶斯
  *
  * 出门堵车的概率就是先验概率，P(堵车)
  * 那么如果我们出门之前我们听到新闻说今天路上出了个交通事故，那么我们想算一下堵车的概率，这个就叫做条件概率，P(堵车|交通事故)
  * 如果我们已经出了门，然后遇到了堵车，那么我们想算一下堵车时由交通事故引起的概率有多大，这是后验概率，也是一种条件概率 P(交通事故|堵车)
  *
  * 那这个就叫做后验概率
  *
  * 优点:
  *   对小规模的数据表现很好，能个处理多分类任务
  *   对缺失数据不太敏感，算法也比较简单，常用于文本分类
  *
  * 缺点:
  *   需要知道先验概率，且先验概率很多时候取决于假设
  *   特征之间需要相互独立
  */
object NaiveBayesDemo extends App {
  spark.sparkContext.setLogLevel("WARN")

  // Load the data stored in LIBSVM format as a DataFrame.
  val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

  // Split the data into training and test sets (30% held out for testing)
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3), seed = 1234L)

  // Train a NaiveBayes model.
  val model = new NaiveBayes()
    .fit(trainingData)

  // Select example rows to display.
  val predictions = model.transform(testData)
  predictions.show()

  // Select (prediction, true label) and compute test error
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println(s"Test set accuracy = $accuracy")
}
