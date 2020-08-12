package com.spark.learning.mllib.demo.features.selector

import org.apache.spark.ml.feature.ChiSqSelector
import org.apache.spark.ml.linalg.Vectors

import com.spark.learning.mllib.demo.spark

/**
  * 卡方选择器
  * numTopFeatures选择固定数量的特征
  * percentile选择部分特征
  * fpr选择p值低于阈值的特征
  * fdr选择错误率低于阈值的特征
  * fwe选择p值低于阈值的所有特征。阈值按 1/numFeatures 进行缩放
  */
object ChiSqSelectorDemo extends App {
  val data = Seq(
    (7, Vectors.dense(0.0, 0.0, 18.0, 1.0), 1.0),
    (8, Vectors.dense(0.0, 1.0, 12.0, 0.0), 0.0),
    (9, Vectors.dense(1.0, 0.0, 15.0, 0.1), 0.0)
  )

  val df = spark.createDataFrame(data).toDF("id", "features", "clicked")

  val selector = new ChiSqSelector()
    .setNumTopFeatures(1)
    .setFeaturesCol("features")
    .setLabelCol("clicked")
    .setOutputCol("selectedFeatures")

  val result = selector.fit(df).transform(df)

  println(s"ChiSqSelector output with top ${selector.getNumTopFeatures} features selected")
  result.show()

}
