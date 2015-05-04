package com.tmp.sgcwaas

import java.io.{OutputStreamWriter, PrintWriter}

import org.apache.spark
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector => MLVector, Vectors}
import org.apache.spark.rdd.RDD
import org.bdgenomics.adam.rdd.ADAMContext._
import org.bdgenomics.formats.avro.{Genotype, _}

import org.bdgenomics.utils.instrumentation.Metrics
import org.bdgenomics.utils.instrumentation.{RecordedMetrics, MetricsListener}
import org.apache.spark.SparkContext
import org.apache.spark.rdd.MetricsContext._
import org.bdgenomics.utils.instrumentation.{Metrics, MetricsListener, RecordedMetrics}

import scala.collection.JavaConverters._
import scala.io.Source

/**
 * From http://bdgenomics.org/blog/2015/02/02/scalable-genomes-clustering-with-adam-and-spark
 */
object Sgcwaas extends spark.Logging {

  def main(args: Array[String]): Unit = {

    if (args.length < 3) {
      System.err.println("Usage: Sgcwaas <master> <variantsfile> <panelfile>")
      System.exit(1)
    }

    setProperties()

    val master = args(0)
    val genotypeFile = args(1)
    val panelFile = args(2)

    val sparkContext = new SparkContext(master, "Sgcwaas")

    Metrics.initialize(sparkContext)
    val metricsListener = new MetricsListener(new RecordedMetrics())
    sparkContext.addSparkListener(metricsListener)

    val sampledGts: RDD[Genotype] = sparkContext.loadGenotypes(genotypeFile)

    logInfo("Loaded genotypes")

    // populations to select
    val pops = Set("GBR", "ASW", "CHB")

    // TRANSFORM THE panelFile Content in the sampleID -> population map
    // containing the populations of interest (pops)
    def extract(filter: (String, String) => Boolean= (s, t) => true) = Source.fromFile(panelFile).getLines().map( line => {
      val toks = line.split("\t").toList
      toks(0) -> toks(1)
    }).toMap.filter( tup => filter(tup._1, tup._2) )

    logInfo("Created population map")

    // panel extract from file, filtering by the 2 populations
    def panel: Map[String,String] =
      extract((sampleID: String, pop: String) => pops.contains(pop))

    // broadcast the panel
    val bPanel = sparkContext.broadcast(panel)

    val finalGts = sampledGts.filter(g =>  bPanel.value.contains(g.getSampleId))

    logInfo("Filtered genotypes")

    // NUMBER OF SAMPLES
    val sampleCount = finalGts.map(_.getSampleId.toString.hashCode).distinct.count
    logInfo(s"#Samples: $sampleCount")

    def variantId(g:Genotype):String = {
      val name = g.getVariant.getContig.getContigName
      val start = g.getVariant.getStart
      val end = g.getVariant.getEnd
      s"$name:$start:$end"
    }
    def asDouble(g:Genotype):Double = g.getAlleles.asScala.count(_ != GenotypeAllele.Ref)

    // A VARIANT SHOULD HAVE sampleCount GENOTYPES
    val variantsById = finalGts.keyBy(g => variantId(g).hashCode).groupByKey
    val missingVariantsRDD = variantsById.filter { case (k, it) => it.size != sampleCount }.keys

    logInfo("Got missing variants")

    // could be broadcasted as well...
    val missingVariants = missingVariantsRDD.collect().toSet

    val completeGts = finalGts.filter { g => ! (missingVariants contains variantId(g).hashCode) }

    logInfo("Got complete genotypes")

    val sampleToData: RDD[(String, (Double, Int))] =
      completeGts.map { g => (g.getSampleId.toString, (asDouble(g), variantId(g).hashCode)) }

    val groupedSampleToData = sampleToData.groupByKey

    logInfo("Grouped data")

    def makeSortedVector(gts: Iterable[(Double, Int)]): MLVector = Vectors.dense( gts.toArray.sortBy(_._2).map(_._1) )

    logInfo("Made ML Vector")

    val dataPerSampleId:RDD[(String, MLVector)] =
      groupedSampleToData.mapValues { it =>
        makeSortedVector(it)
      }

    val dataFrame:RDD[MLVector] = dataPerSampleId.values

    logInfo("Made dataframe")

    val model: KMeansModel = KMeans.train(dataFrame, 3, 10)

    logInfo("Trained model")

    val predictions: RDD[(String, (Int, String))] = dataPerSampleId.map(elt => {
      (elt._1, ( model.predict(elt._2), bPanel.value.getOrElse(elt._1, "")))
    })

    logInfo("Got predictions")

    val confMat = predictions.collect.toMap.values
      .groupBy(_._2).mapValues(_.map(_._1))
      .mapValues(xs => (1 to 3).map( i => xs.count(_ == i-1)).toList)

    logInfo("Computed confusion matrix")

    println("\t1\t2\t3")
    confMat.foreach(entry => {
      println(entry._1 + "\t" + entry._2(0) + "\t" + entry._2(1) + "\t" + entry._2(2))
    })

    val writer = new PrintWriter(new OutputStreamWriter(System.out))
    Metrics.print(writer, Some(metricsListener.metrics.sparkMetrics.stageTimes))
    writer.close()

    logInfo("Finished!")

  }

  def sampleGenotypes(gts: RDD[Genotype]): RDD[Genotype] = {
    val start = 16000000
    val end   = 17000000
    gts.filter(g => (g.getVariant.getStart >= start && g.getVariant.getEnd <= end) )
  }

  def setProperties() = {
    System.setProperty("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    System.setProperty("spark.kryo.registrator", "org.bdgenomics.adam.serialization.ADAMKryoRegistrator")
    System.setProperty("spark.kryoserializer.buffer.mb", "4")
    System.setProperty("spark.kryo.referenceTracking", "true")
  }

}
