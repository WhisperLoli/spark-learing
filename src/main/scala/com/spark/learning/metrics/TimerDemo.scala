package com.spark.learning.metrics
/**
  * Timer度量类型包含了Meter和Histogram的统计，即比率和统计信息的综合
  */
object TimerDemo {
  val time = metricRegistry.timer("requests")

  def main(args: Array[String]): Unit = {
    startReport()

    for (i <- 1 to 5) {
      val context = time.time()
      try {
        run(i)
      } finally {
        context.stop()
      }
    }


    wait5Seconds()
  }

  def run(i: Int): Unit = {
    println(i)
    Thread.sleep(1000)
  }

}
