package com.spark.learning.mllib.demo.regression

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.regression.LinearRegression

/**
  * 线性回归
  */
object LinearRegressionDemo extends App {
  spark.sparkContext.setLogLevel("WARN")

  // Load training data
  val training = spark.read.format("libsvm")
    .load("data/mllib/sample_linear_regression_data.txt")

  val lr = new LinearRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)

  // Fit the model
  val lrModel = lr.fit(training)

  training.show(false)

  // Print the coefficients and intercept for linear regression
  println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

  // Summarize the model over the training set and print out some metrics
  val trainingSummary = lrModel.summary
  println(s"numIterations: ${trainingSummary.totalIterations}")
  println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
  trainingSummary.residuals.show()
  println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
  println(s"r2: ${trainingSummary.r2}")
}
