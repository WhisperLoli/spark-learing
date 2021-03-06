package com.spark.learning.mllib.demo.tuning

import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

/**
  * 交叉验证获取最佳参数
  */
object CrossValidationDemo extends App {
  spark.sparkContext.setLogLevel("WARN")

  val training = spark.createDataFrame(Seq(
    (0L, "a b c d e spark", 1.0),
    (1L, "b d", 0.0),
    (2L, "spark f g h", 1.0),
    (3L, "hadoop mapreduce", 0.0),
    (4L, "b spark who", 1.0),
    (5L, "g d a y", 0.0),
    (6L, "spark fly", 1.0),
    (7L, "was mapreduce", 0.0),
    (8L, "e spark program", 1.0),
    (9L, "a e c l", 0.0),
    (10L, "spark compile", 1.0),
    (11L, "hadoop software", 0.0)
  )).toDF("id", "text", "label")

  // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
  // 切分成单词
  val tokenizer = new Tokenizer()
    .setInputCol("text")
    .setOutputCol("words")
  // 统计词频
  val hashingTF = new HashingTF()
    .setInputCol(tokenizer.getOutputCol)
    .setOutputCol("features")
  // 分类
  val lr = new LogisticRegression()
    .setMaxIter(10)
  val pipeline = new Pipeline()
    .setStages(Array(tokenizer, hashingTF, lr))

  // We use a ParamGridBuilder to construct a grid of parameters to search over.
  // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
  // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
  val paramGrid = new ParamGridBuilder()
    // 测试hashingTF numFeature参数
    .addGrid(hashingTF.numFeatures, Array(10, 100, 1000))
    // 测试lr的regParam参数
    .addGrid(lr.regParam, Array(0.1, 0.01))
    .build()

  // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
  // This will allow us to jointly choose parameters for all Pipeline stages.
  // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
  // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
  // is areaUnderROC.
  val cv = new CrossValidator()
    .setEstimator(pipeline)
    // 设置二分类评估器，LogisticRegression属于二分类
    .setEvaluator(new BinaryClassificationEvaluator)
    .setEstimatorParamMaps(paramGrid)
    .setNumFolds(2)  // Use 3+ in practice
    .setParallelism(2)  // Evaluate up to 2 parameter settings in parallel

  // Run cross-validation, and choose the best set of parameters.
  val cvModel = cv.fit(training)

  // Prepare test documents, which are unlabeled (id, text) tuples.
  val test = spark.createDataFrame(Seq(
    (4L, "spark i j k"),
    (5L, "l m n"),
    (6L, "mapreduce spark"),
    (7L, "apache hadoop")
  )).toDF("id", "text")

  // Make predictions on test documents. cvModel uses the best model found (lrModel).
  cvModel.transform(test)
    .select("id", "text", "probability", "prediction")
    .show(false)
}
