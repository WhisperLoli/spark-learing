package com.spark.learning.metrics

import java.util.concurrent.CopyOnWriteArrayList

import scala.concurrent.Future
import scala.concurrent.ExecutionContext.Implicits.global

/**
  * counter demo:Counter度量类型是一种特殊的Gauge度量，因为其持有的值就是一个AtomicLong，可以递增也可以递减
  * inc递增，dec递减
  */
object CounterDemo {
  // Future使用子线程操作list，普通List并发会出现线程安全问题，所以用CopyOnWriteArrayList
  val list = new CopyOnWriteArrayList[Int]

  val counters = metricRegistry.counter("counters")

  def main(args: Array[String]): Unit = {

    startReport()
    Future {
      run()
    }
    wait5Seconds()
  }

  def run(): Unit = {
    for (i <- 1 to 3) {
      list.add(i)
      counters.inc(1)
      println(list)
      Thread.sleep(1000)
    }

    for (i <- 1 to 3) {
      list.remove(i)
      counters.dec(1)
      println(list)
      Thread.sleep(1000)
    }
  }
}
