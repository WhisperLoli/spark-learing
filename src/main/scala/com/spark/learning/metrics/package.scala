package com.spark.learning

import java.util.concurrent.TimeUnit

import com.codahale.metrics.{ConsoleReporter, MetricRegistry}

package object metrics {
  val metricRegistry = new MetricRegistry

  def startReport() {
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
