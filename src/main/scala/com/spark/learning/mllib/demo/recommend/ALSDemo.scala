package com.spark.learning.mllib.demo.recommend

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS

/**
  * 协同过滤
  */
object ALSDemo extends App {
  spark.sparkContext.setLogLevel("WARN")

  import spark.implicits._

  case class Rating(userId: Int, movieId: Int, rating: Float, timestamp: Long)
  def parseRating(str: String): Rating = {
    val fields = str.split("::")
    assert(fields.size == 4)
    Rating(fields(0).toInt, fields(1).toInt, fields(2).toFloat, fields(3).toLong)
  }

  val ratings = spark.read
    .textFile("data/mllib/als/sample_movielens_ratings.txt")
    .rdd
    .map(parseRating)
    .toDF()

  val Array(training, test) = ratings.randomSplit(Array(0.8, 0.2))

  // Build the recommendation model using ALS on the training data
  val als = new ALS()
    // iterations，迭代的次数
    .setMaxIter(5)
    // 惩罚函数的因数，是ALS的正则化参数
    .setRegParam(0.01)
    .setUserCol("userId")
    .setItemCol("movieId")
    .setRatingCol("rating")
  val model = als.fit(training)

  // Evaluate the model by computing the RMSE on the test data
  // Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
  model.setColdStartStrategy("drop")
  val predictions = model.transform(test)
  predictions.show(false)

  // RMSE 均方根误差，衡量观测值与真实值之间的偏差
  // MSE 均方误差，MSE是真实值与预测值的差值的平方然后求和平均
  val evaluator = new RegressionEvaluator()
    .setMetricName("rmse")
    .setLabelCol("rating")
    .setPredictionCol("prediction")
  val rmse = evaluator.evaluate(predictions)
  println(s"Root-mean-square error = $rmse")

  // Generate top 10 movie recommendations for each user
  val userRecs = model.recommendForAllUsers(10)
  userRecs.show(false)
  // Generate top 10 user recommendations for each movie
  val movieRecs = model.recommendForAllItems(10)
  movieRecs.show(false)

  // Generate top 10 movie recommendations for a specified set of users
  val users = ratings.select(als.getUserCol).distinct().limit(3)
  val userSubsetRecs = model.recommendForUserSubset(users, 10)
  userSubsetRecs.show(false)
  // Generate top 10 user recommendations for a specified set of movies
  val movies = ratings.select(als.getItemCol).distinct().limit(3)
  val movieSubSetRecs = model.recommendForItemSubset(movies, 10)
  movieSubSetRecs.show(false)
}
