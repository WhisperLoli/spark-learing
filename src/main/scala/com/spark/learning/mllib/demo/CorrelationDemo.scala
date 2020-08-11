package com.spark.learning.mllib.demo

import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.sql.Row

/**
  * 相关系数
  *   pearson：两个变量间有线性关系；变量是连续变量；变量均符合正态分布；两变量独立；
  *             实验数据之间的差距不能太大；皮尔森相关性系数受异常值的影响比较大
  *   spearman：比pearson应用更广泛，与数据分布无关
  *
  *   连续数据，正态分布，线性关系，用pearson相关系数是最恰当，
  *   当然用spearman相关系数也可以，就是效率没有pearson相关系数高。
  *   上述任一条件不满足，就用spearman相关系数，不能用pearson相关系数。
  *   两个定序测量数据之间也用spearman相关系数，不能用pearson相关系数
  */
object CorrelationDemo extends App {
  import spark.implicits._
  spark.sparkContext.setLogLevel("WARN")

  val data = Seq(
    // 稀疏矩阵
    Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
    // 密集矩阵
    Vectors.dense(4.0, 5.0, 0.0, 3.0),
    Vectors.dense(6.0, 7.0, 0.0, 8.0),
    Vectors.sparse(4, Seq((0, 9.0), (3, 1.0)))
  )

  val df = data.map(Tuple1.apply).toDF("features")
  df.show()

  // 默认为皮尔森相关系数
  val Row(coeff1: Matrix) = Correlation.corr(df, "features").head
  println(s"Pearson correlation matrix:\n $coeff1")

  // 斯皮尔曼系数
  val Row(coeff2: Matrix) = Correlation.corr(df, "features", "spearman").head
  println(s"Spearman correlation matrix:\n $coeff2")
}
