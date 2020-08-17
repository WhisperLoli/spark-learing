package com.spark.learning.mllib.demo.regression
import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.regression.GeneralizedLinearRegression

/**
  * 广义线性回归
  * 在spark ml中，广义线性回归支持的指数分布分别是正态分布、泊松分布、二项分布以及伽玛分布
  */
object GeneralizedLinearRegressionDemo extends App {
  spark.sparkContext.setLogLevel("WARN")
  // Load training data
  val dataset = spark.read.format("libsvm")
    .load("data/mllib/sample_linear_regression_data.txt")

  // 当指数分布是高斯分布，同时链接函数是恒等(identity)时，此时的情况就是普通的线性回归
  val glr = new GeneralizedLinearRegression()
    .setFamily("gaussian")
    .setLink("identity")
    .setMaxIter(10)
    .setRegParam(0.3)

  // Fit the model
  val model = glr.fit(dataset)

  // Print the coefficients and intercept for generalized linear regression model
  println(s"Coefficients: ${model.coefficients}")
  println(s"Intercept: ${model.intercept}")

  // Summarize the model over the training set and print out some metrics
  val summary = model.summary
  println(s"Coefficient Standard Errors: ${summary.coefficientStandardErrors.mkString(",")}")
  println(s"T Values: ${summary.tValues.mkString(",")}")
  println(s"P Values: ${summary.pValues.mkString(",")}")
  println(s"Dispersion: ${summary.dispersion}")
  println(s"Null Deviance: ${summary.nullDeviance}")
  println(s"Residual Degree Of Freedom Null: ${summary.residualDegreeOfFreedomNull}")
  println(s"Deviance: ${summary.deviance}")
  println(s"Residual Degree Of Freedom: ${summary.residualDegreeOfFreedom}")
  println(s"AIC: ${summary.aic}")
  println("Deviance Residuals: ")
  summary.residuals().show()
}
