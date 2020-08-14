package com.spark.learning.mllib.demo.recommend

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType}

object ALSRecommend extends App {
  spark.sparkContext.setLogLevel("WARN")
  // 当血缘关系过长时，需要设置checkpoint,否则容易出现stackoverflow
  spark.sparkContext.setCheckpointDir("checkpoint")
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
    .withColumn("userRatingR", when($"userRating" === "-1", 3.0d).otherwise($"userRating".cast(DoubleType)))
    .drop("userRating")
    .withColumnRenamed("userRatingR", "userRating")
    .select($"movieId".cast(IntegerType), $"userId", $"userRating".cast(DoubleType))
    .cache()

  df.show()

  val Array(training, test) = df.randomSplit(Array(0.8, 0.2))

  // Build the recommendation model using ALS on the training data
  // 两种正则化方法L1和L2。
  // L2正则化假设模型参数服从高斯分布，L2正则化函数比L1更光滑，所以更容易计算；
  // L1假设模型参数服从拉普拉斯分布，L1正则化具备产生稀疏解的功能
  val als = new ALS()
    // iterations，迭代的次数
    .setMaxIter(20)
    // 惩罚函数的因数，是ALS的正则化参数，L2正则的系数
    .setRegParam(0.01)
    .setUserCol("userId")
    .setItemCol("movieId")
    .setRatingCol("userRating")
    // 并行计算
    .setNumBlocks(10)
    // 表示原始User和Item的rating矩阵的值是否是评判的打分值，False表示是打分值，True表示是矩阵的值是某种偏好
    .setImplicitPrefs(false)
    // coldStartStrategy：String类型。有两个取值"nan" or "drop"。
    // 这个参数指示用在prediction阶段时遇到未知或者新加入的user或item时的处理策略。
    // 尤其是在交叉验证或者生产场景中，遇到没有在训练集中出现的user/item id时。
    // "nan"表示对于未知id的prediction结果为NaN。
    // "drop"表示对于transform()的入参DataFrame中出现未知ids的行，将会在包含prediction的返回DataFrame中被drop。
    // 默认值是"nan"
    .setColdStartStrategy("drop")

  val model = als.fit(training)

  val predictions = model.transform(test)
  predictions.show(false)


  val evaluator = new RegressionEvaluator()
    .setMetricName("rmse")
    .setLabelCol("userRating")
    .setPredictionCol("prediction")
  val rmse = evaluator.evaluate(predictions)
  println(s"Root-mean-square error = $rmse")

  model.recommendForAllItems(10).show()
  model.recommendForAllUsers(3).show()

}
