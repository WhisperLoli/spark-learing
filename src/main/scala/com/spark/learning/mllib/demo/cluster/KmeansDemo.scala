package com.spark.learning.mllib.demo.cluster

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.ClusteringEvaluator

/**
  * K-Means算法的思想很简单，对于给定的样本集，按照样本之间的距离大小，将样本集划分为K个簇。
  * 让簇内的点尽量紧密的连在一起，而让簇间的距离尽量的大
  * K-Means的主要优点有：
  *
  * 　　　　1）原理比较简单，实现也是很容易，收敛速度快。
  * 　　　　2）聚类效果较优。
  * 　　　　3）算法的可解释度比较强。
  * 　　　　4）主要需要调参的参数仅仅是簇数k。
  *
  * K-Means的主要缺点有：
  *
  * 　　　　1）K值的选取不好把握
  * 　　　　2）对于不是凸的数据集比较难收敛
  * 　　　　3）如果各隐含类别的数据不平衡，比如各隐含类别的数据量严重失衡，或者各隐含类别的方差不同，则聚类效果不佳。
  * 　　　　4） 采用迭代方法，得到的结果只是局部最优。
  * 　　　　5） 对噪音和异常点比较的敏感
  */
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
