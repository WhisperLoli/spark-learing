package com.spark.learning.mllib.demo.cluster

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

object KmeansDemo extends App {
  spark.sparkContext.setLogLevel("WARN")

  // Loads data.
  val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")
  dataset.show(false)
  // Trains a k-means model.
  // 分成2类
  val kmeans = new KMeans().setK(2).setSeed(1L)
  val model = kmeans.fit(dataset)

  // Make predictions
  val predictions = model.transform(dataset)
  predictions.show(false)

  // Evaluate clustering by computing Silhouette score
  val evaluator = new ClusteringEvaluator()

  val silhouette = evaluator.evaluate(predictions)
  println(s"Silhouette with squared euclidean distance = $silhouette")

  // 聚类中心点
  println("Cluster Centers: ")
  model.clusterCenters.foreach(println)
}
