package com.spark.learning.akka.demo.robot

sealed trait Action {
  val message: String
  val time: Long
}

case class TurnOnLight(override val time: Long) extends Action {
  // 开灯消息
  val message = "Turn on the living room light"
}

case class BoilWater(override val time: Long) extends Action {
  // 烧水消息
  val message = "Burn a pot of water"
}