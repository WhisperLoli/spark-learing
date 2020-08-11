package com.spark.learning.mllib

import org.apache.spark.sql.SparkSession

/**
  * 机器学习大致分为4类：
  *   分类
  *   回归
  *   聚类
  *   关联分析：分为频繁项集挖掘和关联规则
  */
package object demo {
  val spark = SparkSession
    .builder()
    .appName("spark ml demo")
    .master("local[*]")
    .getOrCreate()
}
