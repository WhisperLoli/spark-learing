package com.spark.learning.mllib.demo.regression

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorIndexer
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}

object GBTRegressorDemo extends App {
  spark.sparkContext.setLogLevel("WARN")
  // Load and parse the data file, converting it to a DataFrame.
  val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")

  // Automatically identify categorical features, and index them.
  // Set maxCategories so features with > 4 distinct values are treated as continuous.
  val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(4)
    .fit(data)

  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

  // Train a GBT model.
  val gbt = new GBTRegressor()
    .setLabelCol("label")
    .setFeaturesCol("indexedFeatures")
    .setMaxIter(10)

  // Chain indexer and GBT in a Pipeline.
  val pipeline = new Pipeline()
    .setStages(Array(featureIndexer, gbt))

  // Train model. This also runs the indexer.
  val model = pipeline.fit(trainingData)

  // Make predictions.
  val predictions = model.transform(testData)

  // Select example rows to display.
  predictions.select("prediction", "label", "features").show(5)

  // Select (prediction, true label) and compute test error.
  val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")
  val rmse = evaluator.evaluate(predictions)
  println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

  val gbtModel = model.stages(1).asInstanceOf[GBTRegressionModel]
  println(s"Learned regression GBT model:\n ${gbtModel.toDebugString}")
}
