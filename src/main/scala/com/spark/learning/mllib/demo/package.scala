package com.spark.learning.mllib

import org.apache.spark.sql.SparkSession

/**
  * 机器学习大致分为4类：
  *   分类
  *   回归
  *   聚类
  *   关联分析：分为频繁项集挖掘和关联规则
  *
  * 如果要执行降维（dimension reduction），则使用主成分分析（PCA）
  * 如果您需要快速进行数值预测（numeric prediction），请使用决策树（decision tree）或逻辑回归（logistic regression）
  * 如果您需要分层结果，则使用分层聚类（hierarchical clustering）
  *
  * 决策树很容易理解和实施。但是，当我们耗尽树枝（branch）并且深入时，它们倾向于过度拟合数据。
  * 随机森林和梯度提升(GBTs)是两种使用树算法的实现，具有良好的精确度，是克服过拟合问题（over-fitting problem）的流行方法
  */
package object demo {
  val spark = SparkSession
    .builder()
    .appName("spark ml demo")
    .master("local[*]")
    .getOrCreate()

  /**
    * 分类：
    *   逻辑回归
    *   SVM
    *   随机森林
    *     bagging派系，各个弱学习器之间没有依赖关系，可以并行拟合，Bagging的子采样是放回采样
    *   GBDTs
    *     boosting派系，各个弱学习器之间有依赖关系，boosting的子采样是无放回采样
    *   XGBoost:
    *     也属于boosting派系，相比GBTs特点就是计算速度快，模型表现好
    *     训练时可以用所有的 CPU 内核来并行化建树
    *     用分布式计算来训练非常大的模型。
    *   朴素贝叶斯
    *   决策树
    *
    * 回归：
    *   线性回归
    *   决策树
    *   随机森林
    *   GBRTs
    *
    * 聚类：
    *   K-Means
    *   高斯混合GMM
    *
    * 关联规则：
    *   FP-growth
    *
    * 协同过滤
    *   基于user, item:
    *     ALS
    *   基于模型:
    *     关联规则
    *     分类
    *     聚类
    *     回归
    */
}
