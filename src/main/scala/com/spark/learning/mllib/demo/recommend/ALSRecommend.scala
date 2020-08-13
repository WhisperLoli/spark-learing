package com.spark.learning.mllib.demo.recommend

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType}

object ALSRecommend extends App {
  import spark.implicits._
  val hotMovie = spark.read.csv("data/mllib/als/hot_movies.csv")
    .toDF("movieId", "rating", "movieName").cache()

  // 建立movieId到moviedName的映射，Map(movieId -> movieName)
  val movieIdAndName = hotMovie.select("movieId", "movieName")
    .rdd
    .mapPartitions(iterator => {
      iterator.map(r => Map(r.getAs[Int](0) -> r.getAs[String](1)))
  }).flatMap(r => r).collectAsMap()

  // println(movieAndName)

  val userMovie = spark.read.csv("data/mllib/als/user_movies.csv")
    .toDF("userName", "movieId", "userRating").cache()

  // 将用户名转换成唯一ID，建立userName到userId的映射
  val userNameToId = userMovie.select("userName")
    .map(r => r.getString(0))
    .rdd.distinct()
    .zipWithUniqueId()
    .collectAsMap()

  // 用户ID到name的映射
  val userIdToName = userNameToId.map(r => (r._2, r._1))

  val columnUserNameToId = udf((userName: String) => {
    userNameToId.get(userName).get
  })

  val df = userMovie.withColumn("userId", columnUserNameToId($"userName"))
    .withColumn("userRating", when($"userRating" === "-1", 3.0d).otherwise($"userRating".cast(DoubleType)))
    .select($"movieId".cast(IntegerType), $"userId", $"userRating".cast(DoubleType))
    .cache()

  df.show()

  val Array(training, test) = df.randomSplit(Array(0.8, 0.2))

  // Build the recommendation model using ALS on the training data
  val als = new ALS()
    // iterations，迭代的次数
    .setMaxIter(20)
    // 惩罚函数的因数，是ALS的正则化参数
    .setRegParam(0.01)
    .setUserCol("userId")
    .setItemCol("movieId")
    .setRatingCol("userRating")

  val model = als.fit(training)

  model.setColdStartStrategy("drop")
  val predictions = model.transform(test)
  predictions.show(false)


  val evaluator = new RegressionEvaluator()
    .setMetricName("rmse")
    .setLabelCol("userRating")
    .setPredictionCol("prediction")
  val rmse = evaluator.evaluate(predictions)
  println(s"Root-mean-square error = $rmse")


}
