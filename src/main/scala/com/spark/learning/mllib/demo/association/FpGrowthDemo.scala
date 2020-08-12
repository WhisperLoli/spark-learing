package com.spark.learning.mllib.demo.association

import com.spark.learning.mllib.demo.spark
import org.apache.log4j.{Level, Logger}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.fpm.FPGrowth

/**
  * 一般来说，支持度高的数据不一定构成频繁项集，但是支持度太低的数据肯定不构成频繁项集
  * Support(AB) = P(AB) = nums(AB)/nums(all)
  *
  * 置信度体现了一个数据出现后，另一个数据出现的概率，或者说数据的条件概率, 𝑃(A|B)=𝑃(AB)/𝑃(B)
  *
  * 在购物数据中，纸巾对应鸡爪的置信度为90%，支持度为1%。
  * 则意味着在购物数据中，总共有1%的用户既买鸡爪又买纸巾;同时买鸡爪的用户中有90%的用户购买纸巾。
  *
  * 频繁项集挖掘：
  *   Apriori算法: 每次扫描数据集寻找项支持度，过滤掉低于MinSupport的项，项值依次增大，k项集依赖k-1项集生成
  *   每增大一次，都需要过滤低于MinSupport的项，每次都需要扫描一次数据集，时间复杂度O(kn)，k为频繁项数
  *
  *   FP-Growth算法: 在构建 FP 树时只需要扫描数据集两次。这种特性使得 FP-Growth 算法比 Aprior 算法速度快。
  *   FP 树是一种前缀树, 时间复杂度O(n)，常数2可以忽略
  *   统计数据集中各个元素出现的频数，将频数小于MinSupport的元素从数据集中删除，将新数据集中的各条记录按项出现频数降序排列
  *   遍历新数据集，构建FP树，再根据FP树构造频繁项集
  *
  * 关联规则：
  *   在频繁项集的基础上发现关联规则，关联规则挖掘首先需要对上文得到的频繁项集构建所有可能的规则，
  *   然后对每条规则逐个计算置信度，输出置信度大于最小置信度的所有规则
  *   下述规则的可信度即为置信度
  *   AB -> C
  *
  */
object FpGrowthDemo extends App with Logging {
  import spark.implicits._

  // 只设置spark的log级别为WARN，其他log级别为info
  Logger.getLogger("org.apache.spark").setLevel(Level.WARN)

  val dataset = spark.createDataset(Seq(
    "1 2 5",
    "1 2 3 5",
    "1 2")
  ).map(t => t.split(" ")).toDF("items").cache()

  val fpgrowth = new FPGrowth()
    .setItemsCol("items")
    // 最小支持度，item数目，支持度就是几个关联的数据在数据集中出现的次数占总数据集的比重
    .setMinSupport(0.6)
    // 最小置信度，当出现某几项item，出现另一项item的置信度要高于设置的值，才会算入频繁项中
    .setMinConfidence(0.6)

  val model = fpgrowth.fit(dataset)

  // 频繁项集
  model.freqItemsets.show()

  // 应用关联规则
  model.associationRules.show()

  // transform examines the input items against all the association rules and summarize the
  // consequents as prediction
  model.transform(dataset).show()

  log.info("finish")
}
