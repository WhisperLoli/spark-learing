package com.spark.learning.metrics

import java.util.concurrent.TimeUnit

import com.codahale.metrics.{ConsoleReporter, Meter, MetricRegistry}

/**
  * meter demo
  */
object MeterDemo {
  def main(args: Array[String]): Unit = {
    startReport(metricRegistry)
    val requests: Meter = metricRegistry.meter("requests")
    requests.mark()
  }

  def startReport(metricRegistry: MetricRegistry) {
    val reporter = ConsoleReporter.forRegistry(metricRegistry)
      .convertRatesTo(TimeUnit.SECONDS)
      .convertDurationsTo(TimeUnit.MILLISECONDS)
      .build()
    reporter.start(1, TimeUnit.SECONDS)
  }

  def wait5Seconds() {
    Thread.sleep(5*1000)
  }
}
