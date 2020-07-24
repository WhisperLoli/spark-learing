package com.spark.learning.akka.demo.robot

import akka.actor.{ActorSystem, Props}

import scala.concurrent.Await
import scala.concurrent.duration._

object RobotApp extends App {
  val actorSystem = ActorSystem("robot-system")
  // 创建一个机器人
  val robotActor = actorSystem.actorOf(Props(classOf[RobotActor]), "robotActor")
  // 给机器人发送一个开灯命令
  robotActor ! TurnOnLight(1)

  // 给机器人发送一个烧水命令
  robotActor ! BoilWater(2)

  // 给机器人发送一个任意命令
  robotActor ! "who are you"

  sys.addShutdownHook({
    actorSystem.terminate()
    Await.result(actorSystem.whenTerminated, 10 seconds)
  })

  Thread.sleep(5000)
  System.exit(0)
}