package com.spark.learning.mllib.demo.classification
import com.spark.learning.mllib.demo.spark
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorIndexer}

/**
  * 决策树分类
  *
  * 决策树的优点：
  *   1. 决策树易于理解和解释；
  *   2. 能够同时处理数据型和类别型属性；
  *   3. 决策树是一个白盒模型，给定一个观察模型，很容易推出相应的逻辑表达式；
  *   4. 在相对较短的时间内能够对大型数据作出效果良好的结果；
  *   5. 比较适合处理有缺失属性值的样本。
  *
  * 决策树的缺点：
  *   1. 对那些各类别数据量不一致的数据，在决策树种，信息增益的结果偏向那些具有更多数值的特征；
  *   2. 容易过拟合；
  *   3. 忽略了数据集中属性之间的相关性。
  */
object DecisionTreeClassifierDemo extends App {
  spark.sparkContext.setLogLevel("WARN")

  // Load the data stored in LIBSVM format as a DataFrame.
  val data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
  data.show()
  // Index labels, adding metadata to the label column.
  // Fit on whole dataset to include all labels in index.
  val labelIndexer = new StringIndexer()
    .setInputCol("label")
    .setOutputCol("indexedLabel")
    .fit(data)
  // Automatically identify categorical features, and index them.
  val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(4) // features with > 4 distinct values are treated as continuous.
    .fit(data)

  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

  // Train a DecisionTree model.
  val dt = new DecisionTreeClassifier()
    .setLabelCol("indexedLabel")
    .setFeaturesCol("indexedFeatures")

  // Convert indexed labels back to original labels.
  val labelConverter = new IndexToString()
    .setInputCol("prediction")
    .setOutputCol("predictedLabel")
    .setLabels(labelIndexer.labels)

  // Chain indexers and tree in a Pipeline.
  val pipeline = new Pipeline()
    .setStages(Array(labelIndexer, featureIndexer, dt, labelConverter))

  // Train model. This also runs the indexers.
  val model = pipeline.fit(trainingData)

  // Make predictions.
  val predictions = model.transform(testData)

  // Select example rows to display.
  predictions.select("predictedLabel", "label", "features").show(5)

  // Select (prediction, true label) and compute test error.
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("indexedLabel")
    .setPredictionCol("prediction")
    .setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println(s"Test Error = ${(1.0 - accuracy)}")

  val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
  println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
}
