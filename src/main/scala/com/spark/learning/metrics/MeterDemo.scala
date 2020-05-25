package com.spark.learning.metrics

import com.codahale.metrics.Meter

/**
  * meter demo：速率（已处理的请求总数除以进程已运行的秒数）
  *
  * 20-5-25 23:38:20 ===============================================================
  * -- Meters ----------------------------------------------------------------------
  * requests
  * count = 1
  * mean rate = 0.99 events/second
  * 1-minute rate = 0.00 events/second
  * 5-minute rate = 0.00 events/second
  * 15-minute rate = 0.00 events/second
  *
  *
  * 20-5-25 23:38:21 ===============================================================
  * -- Meters ----------------------------------------------------------------------
  * requests
  * count = 1
  * mean rate = 0.50 events/second
  * 1-minute rate = 0.00 events/second
  * 5-minute rate = 0.00 events/second
  * 15-minute rate = 0.00 events/second
  *
  *
  * 20-5-25 23:38:22 ===============================================================
  * -- Meters ----------------------------------------------------------------------
  * requests
  * count = 1
  * mean rate = 0.33 events/second
  * 1-minute rate = 0.00 events/second
  * 5-minute rate = 0.00 events/second
  * 15-minute rate = 0.00 events/second
  *
  *
  * 20-5-25 23:38:23 ===============================================================
  * -- Meters ----------------------------------------------------------------------
  * requests
  * count = 1
  * mean rate = 0.25 events/second
  * 1-minute rate = 0.00 events/second
  * 5-minute rate = 0.00 events/second
  * 15-minute rate = 0.00 events/second
  *
  *
  * 20-5-25 23:38:24 ===============================================================
  * -- Meters ----------------------------------------------------------------------
  * requests
  * count = 1
  * mean rate = 0.20 events/second
  * 1-minute rate = 0.00 events/second
  * 5-minute rate = 0.00 events/second
  * 15-minute rate = 0.00 events/second
  */
object MeterDemo {
  def main(args: Array[String]): Unit = {
    startReport()
    val requests: Meter = metricRegistry.meter("requests")
    requests.mark()
    wait5Seconds()
  }
}
