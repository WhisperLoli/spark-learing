package com.spark.learning.mllib.demo.classification
import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.classification.LinearSVC

/**
  * SVM分类
  */
object LinearSupportVectorMachineDemo extends App {
  // Load training data
  val df = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

  val Array(training, testing) = df.randomSplit(Array(0.8, 0.2))

  val lsvc = new LinearSVC()
    .setMaxIter(10)
    .setRegParam(0.1)

  // Fit the model
  val lsvcModel = lsvc.fit(training)

  // Print the coefficients and intercept for linear svc
  println(s"Coefficients: ${lsvcModel.coefficients} Intercept: ${lsvcModel.intercept}")

  lsvcModel.transform(testing).select("label", "prediction").show(false)
}
