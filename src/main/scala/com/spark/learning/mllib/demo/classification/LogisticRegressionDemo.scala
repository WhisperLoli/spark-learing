package com.spark.learning.mllib.demo.classification

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.classification.LogisticRegression
/**
  * Logistic回归
  * 优点：计算代价低，速度快，容易理解和实现。
  * 缺点：容易欠拟合，分类和回归的精度不高
  */
object LogisticRegressionDemo extends App {
  // Load training data
  val training = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

  // 二分类
  val lr = new LogisticRegression()
    .setMaxIter(10)
    // 正则项参数(相当于 λ)
    .setRegParam(0.3)
    // ElasticNet混合参数 α 。如果α=0, 惩罚项是一个L2 penalty。
    // 如果α=1，它是一个L1 penalty。
    // 如果0<α<1，则是L1和L2的结果。缺省为0.0，为L2罚项
    .setElasticNetParam(0.8)

  // Fit the model
  val lrModel = lr.fit(training)

  // Print the coefficients and intercept for logistic regression
  println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")

  // We can also use the multinomial family for binary classification
  // 多分类
  val mlr = new LogisticRegression()
    .setMaxIter(10)
    .setRegParam(0.3)
    .setElasticNetParam(0.8)
    .setFamily("multinomial")

  val mlrModel = mlr.fit(training)

  mlrModel.transform(training).show()

  // Print the coefficients and intercepts for logistic regression with multinomial family
  println(s"Multinomial coefficients: ${mlrModel.coefficientMatrix}")
  println(s"Multinomial intercepts: ${mlrModel.interceptVector}")
}
