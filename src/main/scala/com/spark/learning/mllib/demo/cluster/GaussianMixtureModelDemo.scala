package com.spark.learning.mllib.demo.cluster

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.clustering.GaussianMixture

/**
  * 高斯混合模型
  */
object GaussianMixtureModelDemo extends App {
  val dataset = spark.read.format("libsvm").load("data/mllib/sample_kmeans_data.txt")

  // Trains Gaussian Mixture Model
  val gmm = new GaussianMixture()
    .setK(2)
  val model = gmm.fit(dataset)
  model.transform(dataset).show()

  // output parameters of mixture model model
  for (i <- 0 until model.getK) {
    println(s"Gaussian $i:\nweight=${model.weights(i)}\n" +
      s"mu=${model.gaussians(i).mean}\nsigma=\n${model.gaussians(i).cov}\n")
  }
}
