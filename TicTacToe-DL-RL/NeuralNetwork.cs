using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    public class NeuralNetwork
    {
        public int globalID = -1;
        public int weightsID = -1;
        private Hashtable hashtable = new Hashtable();

        const float softmaxTemperature = 1.0f;

        // for input layer
        public float[] input = new float[GameProperties.INPUT_PLANES * GameProperties.GAMEBOARD_HEIGHT * GameProperties.GAMEBOARD_WIDTH]; // input to complete NN
        public float[] outputConvFilter = new float[Params.FILTERS * GameProperties.GAMEBOARD_HEIGHT * GameProperties.GAMEBOARD_WIDTH];
        public float[] firstConvFilterWeights = new float[Params.FILTERS* GameProperties.INPUT_PLANES * Params.FILTER_HEIGHT * Params.FILTER_WIDTH]; // weights

        // for residual tower
        public float[] inputResidualLayer = new float[Params.FILTERS * GameProperties.GAMEBOARD_HEIGHT * GameProperties.GAMEBOARD_WIDTH]; // input to residual layer
        public float[] outputResidualLayer = new float[Params.FILTERS * GameProperties.GAMEBOARD_HEIGHT * GameProperties.GAMEBOARD_WIDTH];
        public float[] temporary = new float[Params.FILTERS * GameProperties.GAMEBOARD_HEIGHT * GameProperties.GAMEBOARD_WIDTH];
        public float[] convFilterWeights = new float[(Params.CONV_LAYERS-1) * Params.FILTERS * Params.FILTERS * Params.FILTER_HEIGHT * Params.FILTER_WIDTH]; // weights

        // for policy layer
        public float[] convWeightsPolicy = new float[Params.POLICY_FILTERS* Params.FILTERS]; // the filters work on Params.FILTERS input layers, they are 1x1 and there are nofpolicyplanes filters
        public float[] BNMeansPolicy = new float[Params.POLICY_FILTERS]; // weights UNTRAINABLE
        public float[] BNStddevPolicy = new float[Params.POLICY_FILTERS]; // weights UNTRAINABLE
        public float[] BNBetaPolicy = new float[Params.POLICY_FILTERS];
        public float[] BNGammaPolicy = new float[Params.POLICY_FILTERS];
        public float[] policyConnectionWeights = new float[GameProperties.GAMEBOARD_HEIGHT * GameProperties.GAMEBOARD_WIDTH* Params.POLICY_FILTERS * GameProperties.OUTPUT_POLICIES]; // weights
        public float[] policyBiases = new float[GameProperties.OUTPUT_POLICIES]; // weights
        public float[] inputFCLayerPolicy = new float[Params.POLICY_FILTERS * GameProperties.GAMEBOARD_HEIGHT* GameProperties.GAMEBOARD_WIDTH];
        public float[] outputPolicyData = new float[GameProperties.OUTPUT_POLICIES];

        // for value layer
        public float[] convWeightsValue1 = new float[Params.VALUE_FILTERS * Params.FILTERS]; // 1x1 filters, 1 of them for 64 input planes // weights
        public float[] valueConnectionWeights2 = new float[Params.VALUE_HEAD_HIDDEN_LAYER_SIZE]; // weights
        public float[] BNMeansValue = new float[Params.VALUE_FILTERS]; // weights UNTRAINABLE
        public float[] BNStddevValue = new float[Params.VALUE_FILTERS]; // weights UNTRAINABLE
        public float[] BNBetaValue = new float[Params.VALUE_FILTERS];
        public float[] BNGammaValue = new float[Params.VALUE_FILTERS];
        public float[] valueConnectionWeights = new float[GameProperties.GAMEBOARD_HEIGHT * GameProperties.GAMEBOARD_WIDTH*Params.VALUE_FILTERS * Params.VALUE_HEAD_HIDDEN_LAYER_SIZE]; // weights
        public float[] valueBiases = new float[Params.VALUE_HEAD_HIDDEN_LAYER_SIZE]; // weights
        public float[] valueBiasLast = new float[1]; // weights
        public float[] inputFCLayerValue = new float[Params.VALUE_FILTERS *GameProperties.GAMEBOARD_HEIGHT*GameProperties.GAMEBOARD_WIDTH];
        public float[] outputValueData = new float[Params.VALUE_FILTERS * GameProperties.GAMEBOARD_HEIGHT * GameProperties.GAMEBOARD_WIDTH];
        public float[] temporaryValueData = new float[Params.VALUE_HEAD_HIDDEN_LAYER_SIZE];

        // for residual tower + input 
        public float[] BNMeans = new float[Params.CONV_LAYERS * Params.FILTERS]; // UNTRAINABLE
        public float[] BNStddev = new float[Params.CONV_LAYERS * Params.FILTERS]; // UNTRAINABLE

        public float[] BNBetas = new float[Params.CONV_LAYERS * Params.FILTERS];
        public float[] BNGammas = new float[Params.CONV_LAYERS * Params.FILTERS];

        // output of NN
        public float[] softmaxPolicy = new float[GameProperties.OUTPUT_POLICIES];
        public float[] winrateOut = new float[1];
        public float winrateSigOut;

        // complete weights
        public List<float> weights = new List<float>();
        public List<float> untrainable_weights = new List<float>();

        // reader and writer channel to opencl thread
        ChannelReader<Job> reader;
        ChannelWriter<Job> writer;

        public NeuralNetwork()
        {

        }
        /// <summary>
        /// Use this for keras weight, since it doesn't have untrainable params
        /// </summary>
        /// <param name="aWeights"></param>
        public NeuralNetwork(List<float> aWeights)
        {
            weights = new List<float>(aWeights);
            ParseWeightsKeras();
        }
        public NeuralNetwork(List<float> aWeights, List<float> aUntrainableWeights)
        {
            weights = new List<float>(aWeights);
            untrainable_weights = new List<float>(aUntrainableWeights);
            ParseWeights();
        }
        public NeuralNetwork(String file)
        {
            ReadWeightsFromFile(file);
        }
        public void ChannelInit()
        {
            writer = OpenCL.InputChannel.Writer;
            reader = OpenCL.ResponseChannels[globalID].Reader;
        }
        public void EnqueueWeights()
        {
            OpenCL.EnqueueWeights(this);
        }
        public void SetIds(int aWeightsID, int aGlobalID)
        {
            weightsID = aWeightsID;
            globalID = aGlobalID;
        }
        public void DeleteArrays()
        {
            outputConvFilter = null;
            firstConvFilterWeights = null;

            // for residual tower
            inputResidualLayer = null;
            outputResidualLayer = null;
            temporary = null;
            convFilterWeights = null;

            // for policy layer
            convWeightsPolicy = null;
            BNMeansPolicy = null;
            BNStddevPolicy = null;
            BNBetaPolicy = null;
            BNGammaPolicy = null;
            policyConnectionWeights = null;
            policyBiases = null;
            inputFCLayerPolicy = null;
            outputPolicyData = null;

            // for value layer
            convWeightsValue1 = null;
            valueConnectionWeights2 = null;
            BNMeansValue = null;
            BNStddevValue = null;
            BNBetaValue = null;
            BNGammaValue = null;
            valueConnectionWeights = null;
            valueBiases = null;
            valueBiasLast = null;
            inputFCLayerValue = null;
            outputValueData = null;
            temporaryValueData = null;

            // for residual tower + input 
            BNMeans = null;
            BNStddev = null;

            BNBetas = null;
            BNGammas = null;
        }
        public Tuple<float[], float> Predict(GameState pos)
        {
            //if(hashtable[pos] != null)
            //{
            //    return (Tuple<float[], float>)hashtable[pos];
            //}

            // returns array of move evals and V
            /*Not using position history, not using caching*/

            Array.Clear(input,0, input.Length);

            // set nn input
            for (int i = 0; i < GameProperties.BOARD_SIZE_Y; ++i)
            {
                for (int j = 0; j < GameProperties.BOARD_SIZE_X; ++j)
                {
                    if (pos.gameBoard[i, j] == 1)
                    {
                        input[i * GameProperties.BOARD_SIZE_X + j] = 1;
                    }
                    else if (pos.gameBoard[i, j] == -1)
                    {
                        input[GameProperties.BOARD_SIZE_Y * GameProperties.BOARD_SIZE_X + i * GameProperties.BOARD_SIZE_X + j] = 1;
                    }
                }
            }
            for (int i = 0; i < GameProperties.BOARD_SIZE_Y * GameProperties.BOARD_SIZE_X; ++i)
            {   // whose turn it is
                input[GameProperties.BOARD_SIZE_Y * GameProperties.BOARD_SIZE_X * 2 + i] = pos.sideToMove == Player.X ? 1 : 0;
            }
            Tuple<float[], float> resultTuple = ForwardPassCPU(input);
            //hashtable.Add(pos, resultTuple);

            return resultTuple;
        }
        public void PredictGPU(GameState pos)
        {
            /*Not using position history, not using caching*/

            Array.Clear(input, 0, input.Length);

            // set nn input
            for (int i = 0; i < GameProperties.BOARD_SIZE_Y; ++i)
            {
                for (int j = 0; j < GameProperties.BOARD_SIZE_X; ++j)
                {
                    if (pos.gameBoard[i, j] == 1)
                    {
                        input[i * GameProperties.BOARD_SIZE_X + j] = 1;
                    }
                    else if (pos.gameBoard[i, j] == -1)
                    {
                        input[GameProperties.BOARD_SIZE_Y * GameProperties.BOARD_SIZE_X + i * GameProperties.BOARD_SIZE_X + j] = 1;
                    }
                }
            }

            for (int i = 0; i < GameProperties.BOARD_SIZE_Y * GameProperties.BOARD_SIZE_X; ++i)
            {   // whose turn it is
                input[GameProperties.BOARD_SIZE_Y * GameProperties.BOARD_SIZE_X * 2 + i] = pos.sideToMove == Player.X ? 1 : 0;
            }

            Job job = new Job();
            job.input = input.ToList();
            job.globalID = globalID;
            job.weightsID = weightsID;
            writer.TryWrite(job);
        }
        public Tuple<float[], float> GetResultAsync()
        {
            Job job = null;
            bool success = reader.TryRead(out job);
            if (success)
            {
                for (int i = 0; i < 25; ++i)
                {
                    softmaxPolicy[i] = job.output[i];
                }
                winrateSigOut = job.output[25];
                return Tuple.Create(softmaxPolicy, winrateSigOut);
            }
            else
            {
                return null;
            }
        }
        public Tuple<float[], float> GetResultSync()
        {
            Task t = Consume(reader);
            t.Wait();
            return Tuple.Create(softmaxPolicy.ToArray(), winrateSigOut);
        }
        private async Task Consume(ChannelReader<Job> c)
        {
            try
            {
                Job job = await c.ReadAsync();

                for (int i = 0; i < 25; ++i)
                {
                    softmaxPolicy[i] = job.output[i];
                }
                winrateSigOut = job.output[25];
                return;
            }
            catch (ChannelClosedException) { }
        }
        public void CalculateVirtualBNs()
        {
            int BATCHSIZE = 100;
            float[][] intermediateData = new float[BATCHSIZE][];

            Array.Clear(BNMeans, 0, BNMeans.Length);
            Array.Clear(BNStddev, 0, BNStddev.Length);
            Array.Clear(BNMeansPolicy, 0, BNMeansPolicy.Length);
            Array.Clear(BNStddevPolicy, 0, BNStddevPolicy.Length);
            Array.Clear(BNMeansValue, 0, BNMeansValue.Length);
            Array.Clear(BNStddevValue, 0, BNStddevValue.Length);

            // ---------------------------------- input convolution -------------------------------------------

            for (int i = 0; i < BATCHSIZE; ++i)
            {
                /* generate input sample */
                // set nn input
                for (int k = 0; k < GameProperties.BOARD_SIZE_X * GameProperties.BOARD_SIZE_Y; ++k)
                {   // the board itself
                    int rand = RandomGen2.Next(-1, 2);
                    if (rand == 1)
                        input[k] = 1;
                    else if (rand == -1)
                        input[25 + k] = 1;
                }

                int sideToMove = RandomGen2.Next(0, 2);
                for (int k = 0; k < GameProperties.BOARD_SIZE_X * GameProperties.BOARD_SIZE_Y; ++k)
                {   // whose turn it is
                    input[GameProperties.BOARD_SIZE_X * GameProperties.BOARD_SIZE_Y*2 + k] = sideToMove;
                }

                /*Conv layer */
                Convolution(input, outputConvFilter, firstConvFilterWeights, GameProperties.INPUT_PLANES, Params.FILTERS, Params.FILTER_WIDTH, Params.FILTER_HEIGHT, 0);

                /* copy to intermediate */
                intermediateData[i] = new float[outputConvFilter.Length];
                outputConvFilter.CopyTo(intermediateData[i], 0);
            }
            /* calculate the means and stddev..*/
            CalculateBNMeansAndStddev(intermediateData, BNMeans, BNStddev, BATCHSIZE, Params.FILTERS, 0);

            // ..apply means and stddev..
            for (int i = 0; i < BATCHSIZE; ++i)
            {
                BN(intermediateData[i], inputResidualLayer, BNMeans, BNStddev, Params.FILTERS, 0, BNGammas, BNBetas);
                intermediateData[i] = new float[inputResidualLayer.Length];
                inputResidualLayer.CopyTo(intermediateData[i], 0);
            }

            // ---------------------------------- residual tower -------------------------------------------
            // residual tower
            for (int index = 0; index < Params.RES_LAYERS; index += 1)
            {
                float[][] residualSave = new float[BATCHSIZE][];
                for(int i = 0; i < BATCHSIZE; ++i)
                {
                    residualSave[i] = new float[intermediateData[i].Length];
                    intermediateData[i].CopyTo(residualSave[i], 0);
                }

                for (int i = 0; i < BATCHSIZE; ++i)
                {
                    // .. apply the means and stddev..
                    Convolution(intermediateData[i], outputResidualLayer, convFilterWeights, Params.FILTERS, Params.FILTERS, Params.FILTER_WIDTH, Params.FILTER_HEIGHT, index * 2);

                    /* copy to intermediate */
                    intermediateData[i] = new float[outputResidualLayer.Length];
                    outputResidualLayer.CopyTo(intermediateData[i], 0);
                }
                CalculateBNMeansAndStddev(intermediateData, BNMeans, BNStddev, BATCHSIZE, Params.FILTERS, index * 2+1);

                // apply the means and stddev
                for (int i = 0; i < BATCHSIZE; ++i)
                {
                    BN(intermediateData[i], outputResidualLayer, BNMeans, BNStddev, Params.FILTERS, index * 2+1, BNGammas, BNBetas);

                    /* copy to intermediate */
                    intermediateData[i] = new float[outputResidualLayer.Length];
                    outputResidualLayer.CopyTo(intermediateData[i], 0);
                }

                for (int i = 0; i < BATCHSIZE; ++i)
                {
                    Convolution(intermediateData[i], temporary, convFilterWeights, Params.FILTERS, Params.FILTERS, Params.FILTER_WIDTH, Params.FILTER_HEIGHT, index * 2 + 1);

                    /* copy to intermediate */
                    intermediateData[i] = new float[temporary.Length];
                    temporary.CopyTo(intermediateData[i], 0);
                }
                CalculateBNMeansAndStddev(intermediateData, BNMeans, BNStddev, BATCHSIZE, Params.FILTERS, index * 2+2);

                // apply the means and stddev 
                for (int i = 0; i < BATCHSIZE; ++i)
                {
                    BNNoRELU(intermediateData[i], outputResidualLayer, BNMeans, BNStddev, Params.FILTERS, index * 2 +2, BNGammas, BNBetas);
                    AddResidual(outputResidualLayer, outputResidualLayer, residualSave[i]);
                    leakyRELU(outputResidualLayer);

                    /* copy to intermediate */
                    intermediateData[i] = new float[outputResidualLayer.Length];
                    outputResidualLayer.CopyTo(intermediateData[i], 0);
                }
            }

            // ---------------------------------- value head -------------------------------------------
            float[][] intermediateData2 = new float[BATCHSIZE][];
            for (int i = 0; i < BATCHSIZE; ++i)
            {
                Convolution(intermediateData[i], outputValueData, convWeightsValue1, Params.FILTERS, Params.VALUE_FILTERS, 1, 1, 0);

                /* copy to intermediate */
                intermediateData2[i] = new float[outputValueData.Length];
                outputValueData.CopyTo(intermediateData2[i], 0);
            }
            CalculateBNMeansAndStddev(intermediateData2, BNMeansValue, BNStddevValue, BATCHSIZE, Params.VALUE_FILTERS, 0);

            // ---------------------------------- policy head -------------------------------------------
            for (int i = 0; i < BATCHSIZE; ++i)
            {
                Convolution(intermediateData[i], inputFCLayerPolicy, convWeightsPolicy, Params.FILTERS, Params.POLICY_FILTERS, 1, 1, 0);

                /* copy to intermediate */
                intermediateData2[i] = new float[inputFCLayerPolicy.Length];
                inputFCLayerPolicy.CopyTo(intermediateData2[i], 0);
            }
            CalculateBNMeansAndStddev(intermediateData2, BNMeansPolicy, BNStddevPolicy, BATCHSIZE, Params.POLICY_FILTERS, 0);
        }
        private void CalculateBNMeansAndStddev(float[][] intermediateData, float[] BN_means, float[] BN_stddev, int BATCHSIZE, int nofFilters, int index)
        {
            // calc BN means
            for (int i = 0; i < BATCHSIZE; ++i)
            {
                for (int filter = 0; filter < nofFilters; ++filter)
                {
                    //BN_means[index * nofFilters + filter] = 0.0f; // not needed when starting with 0 values
                    for (int k = (intermediateData[i].Length/nofFilters) * filter; k < (intermediateData[i].Length / nofFilters) * (filter+1); k++) 
                    {
                        // read out correct plane,all planes are sequential in intermediatedata
                        BN_means[index * nofFilters + filter] += intermediateData[i][k];
                    }
                }
            }
            for (int filter = 0; filter < nofFilters; ++filter)
            {
                BN_means[index * nofFilters + filter] /= BATCHSIZE* (intermediateData[0].Length / nofFilters);
            }

            // calc BN stddev
            for (int i = 0; i < BATCHSIZE; ++i)
            {
                for (int filter = 0; filter < nofFilters; ++filter)
                {
                    for (int k = (intermediateData[i].Length / nofFilters)*filter; k < (intermediateData[i].Length / nofFilters) * (filter + 1); k++)
                    {
                        BN_stddev[index * nofFilters + filter] += (float)Math.Pow(intermediateData[i][k] - BN_means[index * nofFilters + filter], 2.0);
                    }
                }
            }
            for (int filter = 0; filter < nofFilters; ++filter)
            {
                BN_stddev[index * nofFilters + filter] /= BATCHSIZE * (intermediateData[0].Length / nofFilters);
                BN_stddev[index * nofFilters + filter] = (float)Math.Sqrt(BN_stddev[index * nofFilters + filter] + Params.EPS);
            }
        }
        public Tuple<float[], float> ForwardPassCPU(float[] input)
        {
            /*Conv layer */
            Convolution(input, outputConvFilter, firstConvFilterWeights, GameProperties.INPUT_PLANES, Params.FILTERS, Params.FILTER_WIDTH, Params.FILTER_HEIGHT, 0);
            BN(outputConvFilter, inputResidualLayer, BNMeans, BNStddev, Params.FILTERS, 0, BNGammas, BNBetas);

            /*Residual tower*/
            for (int index = 0; index < Params.RES_LAYERS; index += 1) {
                Convolution(inputResidualLayer, outputResidualLayer, convFilterWeights, Params.FILTERS, Params.FILTERS, Params.FILTER_WIDTH, Params.FILTER_HEIGHT, index*2);
                BN(outputResidualLayer, outputResidualLayer, BNMeans, BNStddev, Params.FILTERS, index*2+1, BNGammas, BNBetas);
                Convolution(outputResidualLayer, temporary, convFilterWeights, Params.FILTERS, Params.FILTERS, Params.FILTER_WIDTH, Params.FILTER_HEIGHT, index*2+1);
                BNWithResidual(temporary, outputResidualLayer, inputResidualLayer, BNMeans, BNStddev, Params.FILTERS, index*2+2, BNGammas, BNBetas);
                
                // temporary holds result
                Array.Copy(outputResidualLayer, 0, inputResidualLayer, 0, outputResidualLayer.Length);
            }

            /*value head*/
            Convolution(inputResidualLayer, outputValueData, convWeightsValue1, Params.FILTERS, Params.VALUE_FILTERS, 1, 1, 0);
            BN(outputValueData, outputValueData, BNMeansValue, BNStddevValue, Params.VALUE_FILTERS, 0, BNGammaValue, BNBetaValue);
            FCLayer(outputValueData, temporaryValueData, valueConnectionWeights, valueBiases,  true); // with rectifier
            FCLayer(temporaryValueData, winrateOut, valueConnectionWeights2, valueBiasLast, false); // 1 output, 1 bias
            winrateSigOut = (float)Math.Tanh(winrateOut[0]);

            /*policy head*/
            Convolution(inputResidualLayer, inputFCLayerPolicy, convWeightsPolicy, Params.FILTERS, Params.POLICY_FILTERS, 1, 1, 0);
            BN(inputFCLayerPolicy, inputFCLayerPolicy, BNMeansPolicy, BNStddevPolicy, Params.POLICY_FILTERS, 0, BNGammaPolicy, BNBetaPolicy);
            FCLayer(inputFCLayerPolicy, outputPolicyData, policyConnectionWeights, policyBiases, false); // without rectifier
            Softmax(outputPolicyData, softmaxPolicy, softmaxTemperature);

            return Tuple.Create(softmaxPolicy, winrateSigOut);
        }
        public void ApplyWeightDecay()
        {
            for(int i = 0; i < weights.Count; ++i)
            {
                weights[i] *= Params.WEIGHT_DECAY_FACTOR;
            }
        }
        private void Convolution(float[] input, float[] output, float[] convWeights,
            int inputPlanes, int nofFilters, int filterWidth, int filterHeight, int index)
        {
            // convolution on GAMEBOARD_WIDTH*GAMEBOARD_HEIGHT*depth
            // with nofFilters filters of filterWidth*filterHeight*inputPlanes size
            // resulting in GAMEBOARD_WIDTH*GAMEBOARD_HEIGHT*x volume
            // zero padding
            // order for conv weights is filter, input channel, height, width == filters, channels, rows, cols ? 
            // order for input is input channel, height, width
            // order for output is filter, height, width
            for (int u = 0; u < output.Length; ++u)
            {
                output[u] = 0.0f;
            }

            for (int j = 0; j < inputPlanes; ++j)
            {
                for (int k = 0; k < GameProperties.GAMEBOARD_HEIGHT; ++k)
                {
                    for (int l = 0; l < GameProperties.GAMEBOARD_WIDTH; ++l)
                    {
                        // looking at a 1x1x1 of the input here, we sum up the 3x3 neighbors (depending on filter size)
                        int x = Math.Max(filterHeight / 2 - k, 0);
                        for (; x < filterHeight; ++x)
                        {
                            for (int y = 0; y < filterWidth; ++y)
                            {
                                for (int i = 0; i < nofFilters; ++i)
                                {
                                    // going through the neighbors
                                    if (k - filterHeight / 2 + x < 0 || k - filterHeight / 2 + x >= GameProperties.GAMEBOARD_HEIGHT ||
                                    l - filterWidth / 2 + y < 0 || l - filterWidth / 2 + y >= GameProperties.GAMEBOARD_WIDTH)
                                    {
                                        // the filter is out of bounds, set to 0 (0 padding)
                                        continue;
                                    }
                                    // when input value is 0 skip all filters
                                    if (input[j * GameProperties.GAMEBOARD_HEIGHT * GameProperties.GAMEBOARD_WIDTH + k * GameProperties.GAMEBOARD_WIDTH + l
                                        + (x - (filterHeight / 2)) * GameProperties.GAMEBOARD_WIDTH + y - (filterWidth / 2)] == 0.0f)
                                    {
                                        break;
                                    }

                                    output[i * GameProperties.GAMEBOARD_HEIGHT * GameProperties.GAMEBOARD_WIDTH + k * GameProperties.GAMEBOARD_WIDTH + l] +=
                                        input[j * GameProperties.GAMEBOARD_HEIGHT * GameProperties.GAMEBOARD_WIDTH + k * GameProperties.GAMEBOARD_WIDTH + l
                                        + (x - (filterHeight / 2)) * GameProperties.GAMEBOARD_WIDTH + y - (filterWidth / 2)] *
                                        convWeights[
                                            index * nofFilters * inputPlanes * filterHeight * filterWidth +
                                            i * inputPlanes * filterHeight * filterWidth +
                                            j * filterHeight * filterWidth +
                                            x * filterWidth + y];
                                }
                            }
                        }
                    }
                }
            }
        }
        private void BN(float[] input, float[] output, float[] BNMeans, float[] BNStdDev, int nofFilters, int index, 
            float[] BNGammas, float[] BNBetas)
        {
            // without residual add
            for (int i = 0; i < nofFilters; ++i)
            {
                // go through each plane coming into BN and apply to each element the means and stddev..
                for (int j = 0; j < input.Length/nofFilters; ++j)
                {
                    // we know the size of one plane by dividing input through number of planes (input.length/noffilters)
                    // batch norm/ batch stddev
                    /* see Alg 1: https://arxiv.org/pdf/1502.03167.pdf */
                    float x_til = (float)((input[i * input.Length / nofFilters + j] - BNMeans[index * nofFilters + i])/
                        (BNStdDev[index * nofFilters + i]));
                    output[i * input.Length / nofFilters + j] = BNGammas[index * nofFilters + i] *x_til+BNBetas[index * nofFilters + i];

                    // relu
                    if (output[i * input.Length / nofFilters + j] < 0.0f)
                        output[i * input.Length / nofFilters + j] = 0.3f * output[i * input.Length / nofFilters + j]; // leaky leakyRELU test
                }
            }
        }
        private void leakyRELU(float[] input)
        {
            for (int i = 0; i < input.Length; ++i)
            {
                // relu
                if (input[i] < 0.0f)
                    input[i] = 0.3f * input[i]; // leaky leakyRELU test
            }
        }
        private void BNNoRELU(float[] input, float[] output, float[] BNMeans, float[] BNStdDev, int nofFilters, int index, float[] BNGammas, float[] BNBetas)
        {
            // without residual add
            for (int i = 0; i < nofFilters; ++i)
            {
                // go through each plane coming into BN and apply to each element the means and stddev..
                for (int j = 0; j < input.Length / nofFilters; ++j)
                {
                    // we know the size of one plane by dividing input through number of planes (input.length/noffilters)
                    // batch norm/ batch stddev
                    /* see Alg 1: https://arxiv.org/pdf/1502.03167.pdf */
                    float x_til = (float)((input[i * input.Length / nofFilters + j] - BNMeans[index * nofFilters + i]) /
                        (BNStdDev[index * nofFilters + i]));
                    output[i * input.Length / nofFilters + j] = BNGammas[index * nofFilters + i] * x_til + BNBetas[index * nofFilters + i];
                }
            }
        }
        private void BNWithResidual(float[] input, float[] output, float[] residual, float[] BNMeans, float[] BNStdDev, int nofFilters, int index, float[] BNGammas, float[] BNBetas)
        {
            for (int i = 0; i < nofFilters; ++i)
            {
                for (int j = 0; j < GameProperties.GAMEBOARD_WIDTH * GameProperties.GAMEBOARD_HEIGHT; ++j)
                {
                    // batch norm/ batch stddev
                    float x_til = (float)((input[i * input.Length / nofFilters + j] - BNMeans[index * nofFilters + i]) /
                        (BNStdDev[index * nofFilters + i]));

                    output[i * input.Length / nofFilters + j] = residual[i * input.Length / nofFilters + j] +
                        BNGammas[index * nofFilters + i] * x_til + BNBetas[index * nofFilters + i];

                    // relu
                    if (output[i * input.Length / nofFilters + j] < 0.0f)
                        output[i * input.Length / nofFilters + j] = 0.3f * output[i * input.Length / nofFilters + j];
                }
            }
        }
        private void AddResidual(float[] input, float[] output, float[] residual)
        {
            for (int i = 0; i < Params.FILTERS; ++i)
            {
                for (int j = 0; j < GameProperties.GAMEBOARD_WIDTH * GameProperties.GAMEBOARD_HEIGHT; ++j)
                {
                    output[i * GameProperties.GAMEBOARD_WIDTH * GameProperties.GAMEBOARD_HEIGHT + j] =
                        input[i * GameProperties.GAMEBOARD_WIDTH * GameProperties.GAMEBOARD_HEIGHT + j] + 
                        residual[i * GameProperties.GAMEBOARD_WIDTH * GameProperties.GAMEBOARD_HEIGHT + j];
                }
            }
        }
        private void FCLayer(float[] input, float[] output, float[] connectionWeights, float[] outputBiases, bool rectifier)
        { 
            // keras uses channels last, so the weights are a bit weirdly distributed here
            for (int i = 0; i < output.Length; ++i)
            {
                output[i] = 0.0f;
                for (int j = 0; j < input.Length; ++j)
                {
                    output[i] += input[j] * connectionWeights[j * output.Length + i];//[i * input.Length + j];//
                }
                output[i] += outputBiases[i];

                if(rectifier && output[i] < 0.0f)
                {
                    output[i] = 0.3f * output[i];
                }
            }
        }
        private void Softmax(float[] input, float[] output, float temperature)
        {
            // must be input length == output length
            float alpha = input.Max();
            alpha /= temperature;

            float denom = 0.0f;
            float[] helper = new float[output.Length];
            for (int i = 0; i < output.Length; i++) {
                float val = (float)Math.Exp((input[i] / temperature) - alpha);
                helper[i] = val;
                denom += val;
            }

            for (int i = 0; i < output.Length; ++i)
            {
                output[i] = helper[i] / denom;
            }
        }
        private void Rectifier(float[] data)
        {
            for (int i = 0; i < data.Length; ++i)
            { 
                // relu
                if (data[i] < 0.0f)
                    data[i] = 0.3f * data[i];
            }
        }
        public void ParseWeights()
        {
            int count = 0;
            int untrainable_count = 0;
            for (int i = 0; i < firstConvFilterWeights.Length; ++i)
            {
                firstConvFilterWeights[i] = weights[count];
                count++;
            }
            for (int i = 0; i < convFilterWeights.Length; ++i)
            {
                convFilterWeights[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNMeans.Length; ++i)
            {
                BNMeans[i] = untrainable_weights[untrainable_count];
                untrainable_count++;
            }
            for (int i = 0; i < BNStddev.Length; ++i)
            {
                BNStddev[i] = untrainable_weights[untrainable_count];
                untrainable_count++;
            }
            for (int i = 0; i < convWeightsPolicy.Length; ++i)
            {
                convWeightsPolicy[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNMeansPolicy.Length; ++i)
            {
                BNMeansPolicy[i] = untrainable_weights[untrainable_count];
                untrainable_count++;
            }
            for (int i = 0; i < BNStddevPolicy.Length; ++i)
            {
                BNStddevPolicy[i] = untrainable_weights[untrainable_count];
                untrainable_count++;
            }
            for (int i = 0; i < policyConnectionWeights.Length; ++i)
            {
                policyConnectionWeights[i] = weights[count];
                count++;
            }
            for (int i = 0; i < policyBiases.Length; ++i)
            {
                policyBiases[i] = weights[count];
                count++;
            }            
            for (int i = 0; i < convWeightsValue1.Length; ++i)
            {
                convWeightsValue1[i] = weights[count];
                count++;
            }         
            for (int i = 0; i < valueConnectionWeights2.Length; ++i)
            {
                valueConnectionWeights2[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNMeansValue.Length; ++i)
            {
                BNMeansValue[i] = untrainable_weights[untrainable_count];
                untrainable_count++;
            }
            for (int i = 0; i < BNStddevValue.Length; ++i)
            {
                BNStddevValue[i] = untrainable_weights[untrainable_count];
                untrainable_count++;
            }
            for (int i = 0; i < valueConnectionWeights.Length; ++i)
            {
                valueConnectionWeights[i] = weights[count];
                count++;
            }
            for (int i = 0; i < valueBiases.Length; ++i)
            {
                valueBiases[i] = weights[count];
                count++;
            }  
            for (int i = 0; i < valueBiasLast.Length; ++i)
            {
                valueBiasLast[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNBetaPolicy.Length; ++i)
            {
                BNBetaPolicy[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNGammaPolicy.Length; ++i)
            {
                BNGammaPolicy[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNBetaValue.Length; ++i)
            {
                BNBetaValue[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNGammaValue.Length; ++i)
            {
                BNGammaValue[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNBetas.Length; ++i)
            {
                BNBetas[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNGammas.Length; ++i)
            {
                BNGammas[i] = weights[count];
                count++;
            }
        }
        public void ParseWeightsKeras()
        {
            int count = 0;
            for (int i = 0; i < firstConvFilterWeights.Length; ++i)
            {
                firstConvFilterWeights[i] = weights[count];
                count++;
            }
            for (int i = 0; i < Params.FILTERS; ++i)
            {
                BNGammas[i] = weights[count];
                count++;
            }
            for (int i = 0; i < Params.FILTERS; ++i)
            {
                BNBetas[i] = weights[count];
                count++;
            }
            for (int i = 0; i < Params.FILTERS; ++i)
            {
                BNMeans[i] = weights[count];
                count++;
            }
            for (int i = 0; i < Params.FILTERS; ++i)
            {
                BNStddev[i] = (float)Math.Sqrt(Params.EPS + weights[count]);
                count++;
            }

            for(int j = 0; j < Params.RES_LAYERS*2; ++j)
            {
                for (int i = 0; i < Params.FILTERS * Params.FILTER_HEIGHT * Params.FILTER_WIDTH * Params.FILTERS; ++i)
                {
                    convFilterWeights[j * Params.FILTERS * Params.FILTERS * Params.FILTER_HEIGHT * Params.FILTER_WIDTH + i] = weights[count];
                    count++;
                }
                for (int i = 0; i < Params.FILTERS; ++i)
                {
                    BNGammas[j*(Params.FILTERS)+i+Params.FILTERS] = weights[count];
                    count++;
                }
                for (int i = 0; i < Params.FILTERS; ++i)
                {
                    BNBetas[j * (Params.FILTERS) + i + Params.FILTERS] = weights[count];
                    count++;
                }
                for (int i = 0; i < Params.FILTERS; ++i)
                {
                    BNMeans[j * (Params.FILTERS) + i + Params.FILTERS] = weights[count];
                    count++;
                }
                for (int i = 0; i < Params.FILTERS; ++i)
                {
                    BNStddev[j * (Params.FILTERS) + i + Params.FILTERS] = (float)Math.Sqrt(Params.EPS + weights[count]);
                    count++;
                }
            }

            for (int i = 0; i < convWeightsValue1.Length; ++i)
            {
                convWeightsValue1[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNGammaValue.Length; ++i)
            {
                BNGammaValue[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNBetaValue.Length; ++i)
            {
                BNBetaValue[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNMeansValue.Length; ++i)
            {
                BNMeansValue[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNStddevValue.Length; ++i)
            {
                BNStddevValue[i] = (float)Math.Sqrt(Params.EPS + weights[count]);
                count++;
            }
            for (int i = 0; i < convWeightsPolicy.Length; ++i)
            {
                convWeightsPolicy[i] = weights[count];
                count++;
            }
            
            for (int i = 0; i < BNGammaPolicy.Length; ++i)
            {
                BNGammaPolicy[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNBetaPolicy.Length; ++i)
            {
                BNBetaPolicy[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNMeansPolicy.Length; ++i)
            {
                BNMeansPolicy[i] = weights[count];
                count++;
            }
            for (int i = 0; i < BNStddevPolicy.Length; ++i)
            {
                BNStddevPolicy[i] = (float)Math.Sqrt(Params.EPS + weights[count]);
                count++;
            }

            for (int i = 0; i < valueConnectionWeights.Length; ++i)
            {
                valueConnectionWeights[i] = weights[count];
                count++;
            }
            for (int i = 0; i < valueBiases.Length; ++i)
            {
                valueBiases[i] = weights[count];
                count++;
            }
            for (int i = 0; i < policyConnectionWeights.Length; ++i)
            {
                policyConnectionWeights[i] = weights[count];
                count++;
            }
            for (int i = 0; i < policyBiases.Length; ++i)
            {
                policyBiases[i] = weights[count];
                count++;
            }

            for (int i = 0; i < valueConnectionWeights2.Length; ++i)
            {
                valueConnectionWeights2[i] = weights[count];
                count++;
            }
            valueBiasLast[0] = weights[count];
        }
        public void ReadWeightsFromFile(string name)
        {
            StreamReader streamReader = new StreamReader(name);
            StreamReader streamReader2 = new StreamReader("untrainable_weights.txt");
            string text = null;
            String[] tokens;

            while ((text = streamReader.ReadLine()) != null)
            {
                tokens = text.Split(' ');
                for (int i = 0; i < tokens.Length; ++i)
                {
                    weights.Add(float.Parse(tokens[i]));
                }
            }
            while ((text = streamReader2.ReadLine()) != null)
            {
                tokens = text.Split(' ');
                for (int i = 0; i < tokens.Length; ++i)
                {
                    untrainable_weights.Add(float.Parse(tokens[i]));
                }
            }
            ParseWeights();
            streamReader.Close();
            streamReader2.Close();
        }
        public void ReadWeightsFromFileKeras(string filename)
        {
            StreamReader streamReader = new StreamReader(filename);
            string text = null;
            String[] tokens;
            weights.Clear();

            while ((text = streamReader.ReadLine()) != null)
            {
                tokens = text.Split();
                for (int i = 0; i < tokens.Length-1; ++i) // ignore last space
                {
                    weights.Add(float.Parse(tokens[i]));
                }
            }
            streamReader.Close();
            ParseWeightsKeras();
        }
        public void SaveWeightsToFile(string name)
        {
            StreamWriter fileWriter = new StreamWriter(name);
            StreamWriter fileWriter2 = new StreamWriter("untrainable_weights.txt");

            for (int i = 0; i < firstConvFilterWeights.Length - 1; ++i)
            {
                fileWriter.Write(firstConvFilterWeights[i] + " ");
            }
            fileWriter.Write(firstConvFilterWeights[firstConvFilterWeights.Length-1] + "\n");

            for (int i = 0; i < convFilterWeights.Length-1; ++i)
            {
                fileWriter.Write(convFilterWeights[i] + " ");
            }
            fileWriter.Write(convFilterWeights[convFilterWeights.Length - 1] + "\n");

            for (int i = 0; i < BNMeans.Length-1; ++i)
            {
                fileWriter2.Write(BNMeans[i] + " ");
            }
            fileWriter2.Write(BNMeans[BNMeans.Length - 1] + "\n");

            //w = 1.0f / std::sqrt(w + epsilon) on read of bn stddivs
            for (int i = 0; i < BNStddev.Length-1; ++i)
            {
                fileWriter2.Write(BNStddev[i] + " ");
            }
            fileWriter2.Write(BNStddev[BNStddev.Length - 1] + "\n");

            // policy head
            for (int i = 0; i < convWeightsPolicy.Length-1; ++i)
            {
                fileWriter.Write(convWeightsPolicy[i] + " ");
            }
            fileWriter.Write(convWeightsPolicy[convWeightsPolicy.Length - 1] + "\n");

            for (int i = 0; i < BNMeansPolicy.Length-1; ++i)
            {
                fileWriter2.Write(BNMeansPolicy[i] + " ");
            }
            fileWriter2.Write(BNMeansPolicy[BNMeansPolicy.Length - 1] + "\n");

            for (int i = 0; i < BNStddevPolicy.Length-1; ++i)
            {
                fileWriter2.Write(BNStddevPolicy[i] + " ");
            }
            fileWriter2.Write(BNStddevPolicy[BNStddevPolicy.Length - 1] + "\n");

            for (int i = 0; i < policyConnectionWeights.Length-1; ++i)
            {
                fileWriter.Write(policyConnectionWeights[i] + " ");
            }
            fileWriter.Write(policyConnectionWeights[policyConnectionWeights.Length - 1] + "\n");

            for (int i = 0; i < policyBiases.Length-1; ++i)
            {
                fileWriter.Write(policyBiases[i] + " ");
            }
            fileWriter.Write(policyBiases[policyBiases.Length - 1] + "\n");

            // for value
            for (int i = 0; i < convWeightsValue1.Length-1; ++i)
            {
                fileWriter.Write(convWeightsValue1[i] + " ");
            }
            fileWriter.Write(convWeightsValue1[convWeightsValue1.Length - 1] + "\n");

            for (int i = 0; i < valueConnectionWeights2.Length-1; ++i)
            {
                fileWriter.Write(valueConnectionWeights2[i] + " ");
            }
            fileWriter.Write(valueConnectionWeights2[valueConnectionWeights2.Length - 1] + "\n");

            for (int i = 0; i < BNMeansValue.Length-1; ++i)
            {
                fileWriter2.Write(BNMeansValue[i] + " ");
            }
            fileWriter2.Write(BNMeansValue[BNMeansValue.Length - 1] + "\n");

            for (int i = 0; i < BNStddevValue.Length-1; ++i)
            {
                fileWriter2.Write(BNStddevValue[i] + " ");
            }
            fileWriter2.Write(BNStddevValue[BNStddevValue.Length - 1] + "\n");

            for (int i = 0; i < valueConnectionWeights.Length-1; ++i)
            {
                fileWriter.Write(valueConnectionWeights[i] + " ");
            }
            fileWriter.Write(valueConnectionWeights[valueConnectionWeights.Length - 1] + "\n");

            for (int i = 0; i < valueBiases.Length-1; ++i)
            {
                fileWriter.Write(valueBiases[i] + " ");
            }
            fileWriter.Write(valueBiases[valueBiases.Length - 1] + "\n");

            for (int i = 0; i < valueBiasLast.Length-1; ++i)
            {
                fileWriter.Write(valueBiasLast[i] + " ");
            }
            fileWriter.Write(valueBiasLast[valueBiasLast.Length - 1] + "\n");
            for (int i = 0; i < BNBetaPolicy.Length - 1; ++i)
            {
                fileWriter.Write(BNBetaPolicy[i] + " ");
            }
            fileWriter.Write(BNBetaPolicy[BNBetaPolicy.Length - 1] + "\n");
            for (int i = 0; i < BNGammaPolicy.Length - 1; ++i)
            {
                fileWriter.Write(BNGammaPolicy[i] + " ");
            }
            fileWriter.Write(BNGammaPolicy[BNGammaPolicy.Length - 1] + "\n");
            for (int i = 0; i < BNBetaValue.Length - 1; ++i)
            {
                fileWriter.Write(BNBetaValue[i] + " ");
            }
            fileWriter.Write(BNBetaValue[BNBetaValue.Length - 1] + "\n");
            for (int i = 0; i < BNGammaValue.Length - 1; ++i)
            {
                fileWriter.Write(BNGammaValue[i] + " ");
            }
            fileWriter.Write(BNGammaValue[BNGammaValue.Length - 1] + "\n");

            for (int i = 0; i < BNBetas.Length -1; ++i)
            {
                fileWriter.Write(BNBetas[i] + " ");
            }
            fileWriter.Write(BNBetas[BNBetas.Length -1] + "\n");

            for (int i = 0; i < BNGammas.Length -1; ++i)
            {
                fileWriter.Write(BNGammas[i] + " ");
            }
            fileWriter.Write(BNGammas[BNGammas.Length -1] + "\n");

            fileWriter.Close();
            fileWriter2.Close();
        }
        public void InitializeWeightsTo0()
        {
            for (int i = 0; i < firstConvFilterWeights.Length; ++i)
            {
                firstConvFilterWeights[i] = 0.0f;
            }
            for (int i = 0; i < convFilterWeights.Length; ++i)
            {
                convFilterWeights[i] = 0.0f;
            }
            for (int i = 0; i < BNMeans.Length; ++i)
            {
                BNMeans[i] = 0.0f;
            }
            for (int i = 0; i < BNStddev.Length; ++i)
            {
                BNStddev[i] = 1.0f;
            }
            for (int i = 0; i < convWeightsPolicy.Length; ++i)
            {
                convWeightsPolicy[i] = 0.0f;
            }
            for (int i = 0; i < BNMeansPolicy.Length; ++i)
            {
                BNMeansPolicy[i] = 0.0f;
            }
            for (int i = 0; i < BNStddevPolicy.Length; ++i)
            {
                BNStddevPolicy[i] = 1.0f;
            }
            for (int i = 0; i < policyConnectionWeights.Length; ++i)
            {
                policyConnectionWeights[i] = 0.0f;
            }
            for (int i = 0; i < policyBiases.Length; ++i)
            {
                policyBiases[i] = 0.0f;
            }
            for (int i = 0; i < convWeightsValue1.Length; ++i)
            {
                convWeightsValue1[i] = 0.0f;
            }
            for (int i = 0; i < valueConnectionWeights2.Length; ++i)
            {
                valueConnectionWeights2[i] = 0.0f;
            }
            for (int i = 0; i < BNMeansValue.Length; ++i)
            {
                BNMeansValue[i] = 0.0f;
            }
            for (int i = 0; i < BNStddevValue.Length; ++i)
            {
                BNStddevValue[i] = 1.0f;
            }
            for (int i = 0; i < valueConnectionWeights.Length; ++i)
            {
                valueConnectionWeights[i] = 0.0f;
            }
            for (int i = 0; i < valueBiases.Length; ++i)
            {
                valueBiases[i] = 0.0f;
            }
            for (int i = 0; i < valueBiasLast.Length; ++i)
            {
                valueBiasLast[i] = 0.0f;
            }
            for (int i = 0; i < BNBetaPolicy.Length; ++i)
            {
                BNBetaPolicy[i] = 0.0f;
            }
            for (int i = 0; i < BNGammaPolicy.Length; ++i)
            {
                BNGammaPolicy[i] = 1.0f;
            }
            for (int i = 0; i < BNBetaValue.Length; ++i)
            {
                BNBetaValue[i] = 0.0f;
            }
            for (int i = 0; i < BNGammaValue.Length; ++i)
            {
                BNGammaValue[i] = 1.0f;
            }

            for (int i = 0; i < BNBetas.Length; ++i)
            {
                BNBetas[i] = 0.0f;
                BNGammas[i] = 1.0f;
            }
        }
        public void InitializeWeights(float gaussianSquashFactor)
        {
            for (int i = 0; i < firstConvFilterWeights.Length; ++i)
            {
                firstConvFilterWeights[i] = RandomNr.GetGaussianFloat()*gaussianSquashFactor;
            }
            for (int i = 0; i < convFilterWeights.Length; ++i)
            {
                convFilterWeights[i] = RandomNr.GetGaussianFloat()*gaussianSquashFactor;
            }
            for (int i = 0; i < BNMeans.Length; ++i)
            {
                BNMeans[i] = 0.0f;
            }
            for (int i = 0; i < BNStddev.Length; ++i)
            {
                BNStddev[i] = 1.0f;
            }
            for (int i = 0; i < convWeightsPolicy.Length; ++i)
            {
                convWeightsPolicy[i] = RandomNr.GetGaussianFloat()*gaussianSquashFactor;
            }
            for (int i = 0; i < BNMeansPolicy.Length; ++i)
            {
                BNMeansPolicy[i] = 0.0f;
            }
            for (int i = 0; i < BNStddevPolicy.Length; ++i)
            {
                BNStddevPolicy[i] = 1.0f;
            }
            for (int i = 0; i < policyConnectionWeights.Length; ++i)
            {
                policyConnectionWeights[i] = RandomNr.GetGaussianFloat()*gaussianSquashFactor;
            }
            for (int i = 0; i < policyBiases.Length; ++i)
            {
                policyBiases[i] = RandomNr.GetGaussianFloat()*gaussianSquashFactor;
            }
            for (int i = 0; i < convWeightsValue1.Length; ++i)
            {
                convWeightsValue1[i] = RandomNr.GetGaussianFloat()*gaussianSquashFactor;
            }
            for (int i = 0; i < valueConnectionWeights2.Length; ++i)
            {
                valueConnectionWeights2[i] = RandomNr.GetGaussianFloat()*gaussianSquashFactor;
            }
            for (int i = 0; i < BNMeansValue.Length; ++i)
            {
                BNMeansValue[i] = 0.0f;
            }
            for (int i = 0; i < BNStddevValue.Length; ++i)
            {
                BNStddevValue[i] = 1.0f;
            }
            for (int i = 0; i < valueConnectionWeights.Length; ++i)
            {
                valueConnectionWeights[i] = RandomNr.GetGaussianFloat()*gaussianSquashFactor;
            }
            for (int i = 0; i < valueBiases.Length; ++i)
            {
                valueBiases[i] = RandomNr.GetGaussianFloat()*gaussianSquashFactor;
            }
            for (int i = 0; i < valueBiasLast.Length; ++i)
            {
                valueBiasLast[i] = RandomNr.GetGaussianFloat()*gaussianSquashFactor;
            }
            for (int i = 0; i < BNBetaPolicy.Length; ++i)
            {
                BNBetaPolicy[i] = 0.0f;
            }
            for (int i = 0; i < BNGammaPolicy.Length; ++i)
            {
                BNGammaPolicy[i] = 1.0f;
            }
            for (int i = 0; i < BNBetaValue.Length; ++i)
            {
                BNBetaValue[i] = 0.0f;
            }
            for (int i = 0; i < BNGammaValue.Length; ++i)
            {
                BNGammaValue[i] = 1.0f;
            }
            for (int i = 0; i < BNBetas.Length; ++i)
            {
                BNBetas[i] = 0.0f;
                BNGammas[i] = 1.0f;
            }
        }
    }
}
 
 