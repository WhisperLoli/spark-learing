package com.spark.learning.future

import scala.concurrent.{Await, Future, Promise}
import scala.concurrent.ExecutionContext.Implicits.global
import scala.concurrent.duration._
import scala.util.{Failure, Success}



/**
  * learning Future，启用多线程执行，顺序不能保证，如果主线程执行完，停止任务，可能子线程还没进行
  */
object FutureLearning {
  def main(args: Array[String]): Unit = {
    val result = Future{
      Thread.sleep(10000)
      1
    }

    // 最长等待9秒，线程睡眠10秒，等待超时，TimeoutException
    // Await.result(result, 9 seconds)
    Await.result(result, atMost = 10 seconds)
    if (result.isCompleted) {
      result.onComplete {
        case Success(value) => println(s"current value is  $value")
        case Failure(error) => println(error.getMessage)
      }

      result.onSuccess {
        case value => println(value)
      }
    } else {
      println("result is not finished")
    }

    // 案例二，与Promise一起使用
    case class TaxCut(reduction: Int)

    val taxCut = Promise[TaxCut]()

    // taxCut被赋值后会直接走回调，如果没赋值，主线程结束就不会运行
    taxCut.future.onComplete {
      case Success(value) => println(s"start current taxCut is $value")
      case Failure(error) => println(error)
    }

    // taxCut.success(TaxCut(20))

    // 判断当前是否赋值，赋值后再回调
    if (taxCut.future.isCompleted) {
      taxCut.future.onComplete {
        case Success(value) => println(s"current taxCut is $value")
        case Failure(error) => println(error)
      }
    } else {
      println("taxCut is not finished")
    }

    println(s"可用线程数 ${Runtime.getRuntime.availableProcessors()}")
    Thread.sleep(10000)
  }
}

