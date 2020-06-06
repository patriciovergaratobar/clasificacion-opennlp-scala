import java.io.IOException
import java.io.File

import opennlp.tools.doccat.{DoccatFactory, DoccatModel, DocumentCategorizerME, DocumentSample, DocumentSampleStream}
import opennlp.tools.ml.AbstractTrainer
import opennlp.tools.ml.naivebayes.NaiveBayesTrainer
import opennlp.tools.ml.perceptron.PerceptronTrainer
import opennlp.tools.util.{ObjectStream, PlainTextByLineStream}
import opennlp.tools.util.MarkableFileInputStreamFactory
import opennlp.tools.util.TrainingParameters

import scala.util.{Failure, Success, Try}

object ClasificacionComentarios {

  def main(args: Array[String]): Unit = {

    val archivoEntrenamiento = "src/main/resources/textos.train"
    val modeloEntrenado = entrenarModelo(archivoEntrenamiento)
    val clasificador = clasificar(modeloEntrenado)

    imprimirCategoria(clasificador("No me gusta el pizza!"))
    imprimirCategoria(clasificador("Me encanta la vida que tengo!"))
    imprimirCategoria(clasificador("No me gusta el clima!"))
    imprimirCategoria(clasificador("Que buen clima hay en Villarica!"))
    imprimirCategoria(clasificador("Super bueno el trabajo que realizaste!"))
    imprimirCategoria(clasificador("Es mala la cerveza cuando no esta helada!"))
    imprimirCategoria(clasificador("Agradezco la ayuda que me dierón mis compañeros"))
    imprimirCategoria(clasificador("La aplicación no compila!"))
    imprimirCategoria(clasificador("Rompiste los juguetes!"))
    imprimirCategoria(clasificador("El servicio de VTR es vergonzoso"))
    imprimirCategoria(clasificador("Que bueno que llego el encargo"))
    imprimirCategoria(clasificador("Que mal que todabia no envian la encomienda"))
  }

  /**
   * Funcion que realiza el entrenamiento y entrega un modelo entrenado.
   * @param archivoEntrenamiento
   * @return
   */
  def entrenarModelo(archivoEntrenamiento: String): Option[DoccatModel] = Try({

    val datos = new MarkableFileInputStreamFactory(new File(archivoEntrenamiento))
    val lineStream = new PlainTextByLineStream( datos, "UTF-8")
    val ejemplosStream: ObjectStream[DocumentSample] = new DocumentSampleStream(lineStream)
    //Se especifica el número mínimo de veces que se debe ver una característica
    val parametroEntrenamiento = new TrainingParameters
    parametroEntrenamiento.put(TrainingParameters.ITERATIONS_PARAM, 10)
    parametroEntrenamiento.put(TrainingParameters.CUTOFF_PARAM, 1)
    parametroEntrenamiento.put(AbstractTrainer.ALGORITHM_PARAM, NaiveBayesTrainer.NAIVE_BAYES_VALUE)
    //Se procesa el entrenamiento

    DocumentCategorizerME.train("es", ejemplosStream,parametroEntrenamiento, new DoccatFactory)
  })  match {
    case Success(modeloEntrenado) => Some(modeloEntrenado)
    case Failure(e: IOException) =>
      println(e)
      None
    case _ => None
  }

  /**
   * Funcion que se encarga de clasificar un texto en base al modelo entrenado.
   * @param modeloEntrenado
   * @return
   */
  def clasificar(modeloEntrenado: Option[DoccatModel]) = (valor: String) => modeloEntrenado match {
    case Some(modelo) =>
      val categorizador = new DocumentCategorizerME(modelo)
      val doc = valor.replaceAll("[^A-Za-z]", " ").split(" ")

      val resultados = categorizador.categorize(doc)
      resultados.foreach(print)
      print(s" | $valor")
      val categoria = categorizador.getBestCategory(resultados)
      categoria
    case None => "-1"
  }

  /**
   * Funcion que imprime la clasificacion de cada texto.
   * @param categoria
   */
  def imprimirCategoria(categoria: String): Unit = categoria match {
    case "-1" => println(" | No clasificado.")
    case "0" => println(" | Comentario Negativo.")
    case "1" => println(" | Comentario Positivo.")
    case _ => println(" | No identificado.")
  }
}
