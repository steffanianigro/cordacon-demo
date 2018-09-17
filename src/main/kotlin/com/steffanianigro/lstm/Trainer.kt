package com.steffanianigro.lstm

import org.apache.log4j.BasicConfigurator
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.LSTM
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.learning.config.RmsProp
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction
import org.deeplearning4j.util.ModelSerializer
import java.io.*
import java.util.*


/**
 * This example trains a LSTM RNN on time series data.
 *
 * This example has not been optimised and should NOT be used in a production setting.
 *
 * @author Steffan Ianigro
 */

object BasicRNNExample {

    // RNN dimensions
    private val INPUT_LAYER_WIDTH = 1
    private val HIDDEN_LAYER_WIDTH = 50
    private val OUTPUT_LAYER_WIDTH = 1
    private val HIDDEN_LAYER_CONT = 2

    private val NUM_ITERATIONS = 1

    private fun encodeBase64(file: String) : String {
        val originalFile = File(file)
        var encodedBase64: String? = null
        try {
            val fileInputStreamReader = FileInputStream(originalFile)
            val bytes = ByteArray(originalFile.length().toInt())
            fileInputStreamReader.read(bytes)
            encodedBase64 = String(Base64.getEncoder().encode(bytes))
        } catch (e: FileNotFoundException) {
            e.printStackTrace()
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return encodedBase64 ?: throw IllegalArgumentException("Could not encode base 64.")
    }

    private fun decodeBase64(base64: String) : InputStream {
        var inputStream: InputStream? = null
        try {
            val bytes = Base64.getDecoder().decode(base64)
            inputStream = ByteArrayInputStream(bytes)
        } catch (e: FileNotFoundException) {
            e.printStackTrace()
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return inputStream ?: throw IllegalArgumentException("Could not decode base 64.")
    }

    @JvmStatic
    fun main(args: Array<String>) {

        val uiServer = UIServer.getInstance()
        val statsStorage = InMemoryStatsStorage()
        uiServer.attach(statsStorage)
        var dataPoints = listOf(0.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f)
        // Normalise endpoints between -1 and 1 (same as TANH Activation output)
        val normalizedDataPoints = mutableListOf<Float>()
        val min = dataPoints.min() ?: throw IllegalArgumentException("Could not get min.")
        val range = (dataPoints.max() ?: throw IllegalArgumentException("Could not get max.")) - min
        dataPoints.forEach{ performance ->
            val normalized = (performance - min) / range
            normalizedDataPoints.add((normalized * 2f - 1f))
        }
        println(normalizedDataPoints)
        // Common parameters to setup LSTM
        val builder = NeuralNetConfiguration.Builder()
        builder.seed(123)
        builder.biasInit(0.0)
        builder.miniBatch(false)
        builder.updater(RmsProp(0.0001))
        builder.weightInit(WeightInit.XAVIER)

        val listBuilder = builder.list()

        // Use LSTM builder to construct network
        for (i in 0 until HIDDEN_LAYER_CONT) {
            val hiddenLayerBuilder = LSTM.Builder()
            // If first layer, neuron inputs should be same as number of time series. In this case 1.
            hiddenLayerBuilder.nIn(if (i == 0) INPUT_LAYER_WIDTH else HIDDEN_LAYER_WIDTH)
            hiddenLayerBuilder.nOut(HIDDEN_LAYER_WIDTH)
            // Use TANH activation function which is common for LSTMs
            hiddenLayerBuilder.activation(Activation.TANH)
            listBuilder.layer(i, hiddenLayerBuilder.build())
        }

        // We need to use RnnOutputLayer for our RNN. Loss function is MSE which is good combined with IDENTITY Activation.
        val outputLayerBuilder = RnnOutputLayer.Builder(LossFunction.MSE)
        outputLayerBuilder.activation(Activation.IDENTITY)
        outputLayerBuilder.nIn(HIDDEN_LAYER_WIDTH)

        // Output should be the same as number of time series arrays. In this case, 1.
        outputLayerBuilder.nOut(OUTPUT_LAYER_WIDTH)
        listBuilder.layer(HIDDEN_LAYER_CONT, outputLayerBuilder.build())

        // Finish builder
        listBuilder.pretrain(false)
        listBuilder.backprop(true)

        // Create network
        val conf = listBuilder.build()
        val net = MultiLayerNetwork(conf)
        net.init()
        net.setListeners(StatsListener(statsStorage))

        // CREATE OUR TRAINING DATA
        // The LSTM has one sequence with one input node with a set size equivalent to the length of the input time series data - 1.
        // Output would be the value at the input value index element plus one.
        val setSize = normalizedDataPoints.size - 1
        val input = Nd4j.zeros(NUM_ITERATIONS, 1, setSize)
        val labels = Nd4j.zeros(NUM_ITERATIONS, 1, setSize)
        for (iteration in 0 until NUM_ITERATIONS) {
            // Add all existing data points
            normalizedDataPoints.forEachIndexed { index, currentValue ->
                if (index != normalizedDataPoints.size - 1) {
                    // Inout should be triggering value.
                    input.putScalar(intArrayOf(0, 0, index), currentValue)
                    // Label is desired value of ANN.
                    labels.putScalar(intArrayOf(0, 0, index), normalizedDataPoints[index + 1])
                }
            }
        }
        // Create training data from input and labels (expected output)
        val trainingData = DataSet(input, labels)
        println(trainingData)
        // Create test inputs
        val testInputs = mutableListOf<Float>()
        var actualOutput = 0f
        val finalIndex = setSize - 1
        // Add all existing data points
        normalizedDataPoints.forEachIndexed { index, dataPoint ->
            if (index != normalizedDataPoints.size - 1) {
                testInputs.add(dataPoint)
            } else {
                actualOutput = dataPoint
            }
        }
        val testInput = Nd4j.zeros(1, 1, testInputs.size)
        testInputs.forEachIndexed{ index, testDataPoint ->
            testInput.putScalar(intArrayOf(0, 0, index), testDataPoint)
        }
        // Train the model fro 350 epocs
        for (epoch in 0..350) {
            net.fit(trainingData)
            // Reset RNN
            net.rnnClearPreviousState()
            val output = net.rnnTimeStep(testInput)
            val loss = Math.abs(actualOutput - output.getFloat(finalIndex.toLong()))
            println("Loss: $loss Guess ${output.getFloat(finalIndex.toLong())} Actual $actualOutput")
        }
        // Save the model which can be stored on ledger any way you wish.
        val locationToSave = File("LSTM.zip") //Where to save the network. Note: the file is in .zip format - can be opened externally
        val saveUpdater = true //Updater: i.e., the state for Momentum, RMSProp, Adagrad etc. Save this if you want to train your network more in the future
        ModelSerializer.writeModel(net, locationToSave, saveUpdater)

        // Example of construction base 64 from ZIP file.
        val base64Network = encodeBase64("LSTM.zip")

        // Example of restoring net from base 64. This can be done on ledger for validation. You can also pass in path of ZIP file to restore net as well as a few other methods.
        val restoredNet = ModelSerializer.restoreMultiLayerNetwork(decodeBase64(base64Network))
        restoredNet.rnnClearPreviousState()
        val output = restoredNet.rnnTimeStep(testInput)
        val loss = Math.abs(actualOutput - output.getFloat(finalIndex.toLong()))
        println("Restored loss $loss")
    }
}

fun main(args: Array<String>) {
    BasicConfigurator.configure()
    BasicRNNExample.main(args)
}