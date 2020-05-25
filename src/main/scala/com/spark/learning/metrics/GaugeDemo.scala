package com.spark.learning.metrics

import java.util.concurrent.CopyOnWriteArrayList

import com.codahale.metrics.Gauge

import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

/**
  * gauge demo: 返回一个变量的瞬时值
  */
object GaugeDemo {
  // Future使用子线程操作list，普通List并发会出现线程安全问题，所以用CopyOnWriteArrayList
  val list = new CopyOnWriteArrayList[Int]

  def main(args: Array[String]): Unit = {
    startReport()
    Future {
      run()
    }
    val requests = metricRegistry.register("requests", new Gauge[Int](){
      override def getValue: Int = list.size()
    })
    requests.getValue
    wait5Seconds()
  }

  def inputElement(i: Int): Unit = {
    GaugeDemo.list.add(i)
  }

  def run() {
    println("gauge running")

    for (i <- 1 to 5) {
      inputElement(i)
      println(list)
      Thread.sleep(1000)
    }
  }
}