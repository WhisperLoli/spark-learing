package com.spark.learning.metrics

import scala.concurrent.Future
import scala.util.Random
import scala.concurrent.ExecutionContext.Implicits.global

/**
  * Histogram度量类型用于测量一个数据流各值的统计分布。
  * 其除了能够测量最大值、最小值、平均值外，还可以测量中位数、75、90、95、98、99和99.9%等。
  */
object HistogramDemo {
  val requests = metricRegistry.histogram("requests")

  def main(args: Array[String]): Unit = {
    startReport()
    Future {
      run()
    }
    wait5Seconds()
  }

  def run(): Unit = {
    for (i <- 1 to 5) {
      requests.update(randomNumber())
      Thread.sleep(1000)
    }
  }

  def randomNumber(): Int = {
    val random = new Random()
    random.nextInt(100)
  }
}
