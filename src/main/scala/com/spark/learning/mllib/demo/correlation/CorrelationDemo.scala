package com.spark.learning.mllib.demo.correlation

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.linalg.{Matrix, Vectors}
import org.apache.spark.ml.stat.{ChiSquareTest, Correlation}
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
  *
  *   假设检验是一种强大的统计工具，可用来确定结果是否具有统计学意义，以及该结果是否偶然发生。
  *   spark.ml目前支持Pearson的卡方校验是否具有独立性
  *   卡方检验常用于特征筛选
  */
object CorrelationDemo extends App {
  import spark.implicits._
  spark.sparkContext.setLogLevel("WARN")

  // 一个稠密向量通过一个double类型的数组保存数据，这个数组表示向量的条目值(entry values)；
  // 一个稀疏向量通过两个并行的数组（indices和values）保存数据。
  // 例如，一个向量(1.0, 0.0, 3.0)可以以稠密的格式保存为[1.0, 0.0, 3.0]
  // 或者以稀疏的格式保存为(3, [0, 2], [1.0, 3.0])，其中3表示数组的大小
  val data = Seq(
    // 稀疏向量，(0, 1.0)代表下标为0的值为1.0, （3，-2.0)代表下标为3的值为-2。0
    Vectors.sparse(4, Seq((0, 1.0), (3, -2.0))),
    // 密集向量
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

  // 假设检验数据集
  val data_v = Seq(
    (0.0, Vectors.dense(0.5, 10.0)),
    (0.0, Vectors.dense(1.5, 20.0)),
    (1.0, Vectors.dense(1.5, 30.0)),
    (0.0, Vectors.dense(3.5, 30.0)),
    (0.0, Vectors.dense(3.5, 40.0)),
    (1.0, Vectors.dense(3.5, 40.0))
  )

  // 假设检验，ChiSquare也被称为卡方分布
  val df_v = data_v.toDF("label", "features")
  val chi = ChiSquareTest.test(df_v, "features", "label")
  chi.show()
  // pValue：统计学根据显著性检验方法所得到的P 值。
  // 一般以P < 0.05 为显著， P<0.01 为非常显著，其含义是样本间的差异由抽样误差所致的概率小于0.05 或0.01。
  // 一般来说，假设检验主要看P值就够了。在本例中pValue = 0.68，说明差别无显著意义
  // P值非常小，说明可以拒绝“某列与标签列无关”的假设。也就是说，可以认为每一列的数据都与最后的标签有相关性
  println(s"pValues = ${chi.head().getAs[Vector[String]](0)}")
  // degrees of freedom：自由度。表示可自由变动的样本观测值的数目，
  println(s"degreesOfFreedom ${chi.head().getSeq[Int](1).mkString("[", ",", "]")}")
  // statistic： 检验统计量。简单来说就是用来决定是否可以拒绝原假设的证据。
  // 检验统计量的值是利用样本数据计算得到的，它代表了样本中的信息。
  // 检验统计量的绝对值越大，拒绝原假设的理由越充分，反之，不拒绝原假设的理由越充分
  println(s"statistics ${chi.head().getAs[Vector[String]](2)}")
}
