package com.tmp.sgcwaas

import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkContext}
import org.bdgenomics.adam.rdd.ADAMContext._
import org.bdgenomics.formats.avro.Genotype

/**
 * From http://bdgenomics.org/blog/2015/02/02/scalable-genomes-clustering-with-adam-and-spark
 */
class SampleGenotypes extends Logging {

  def main(args: Array[String]): Unit = {

    if (args.length < 3) {
      System.err.println("Usage: Sgcwaas <master> <variantsfile> <panelfile>")
      System.exit(1)
    }

    setProperties()

    val master = args(0)
    val genotypeFile = args(1)

    val sparkContext = new SparkContext(master, "Sgcwaas")

    val gts: RDD[Genotype] = sparkContext.loadGenotypes(genotypeFile)

    logInfo("Loaded genotypes")

    val sampledGts: RDD[Genotype] = sampleGenotypes(gts)

    logInfo("Sampled genotypes")

    sampledGts.adamParquetSave(genotypeFile + ".sampled.adam")

    logInfo("Saved sampled genotypes")

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
