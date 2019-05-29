using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    class NeuralNetwork
    {
        public bool GPU_PREDICT = false;
        public int globalID = -1;

        // params
        const int gameboardWidth = 5;
        const int gameboardHeight = 5;
        const int filterWidth = 3;
        const int filterHeight = 3;
        const int nofPlanes = 2; // = input channels, 1 plane is board 5x5 + 1 plane color 5x5
        const int nofOutputPolicies = 25; // policy net has 25 outputs (1 per potential move)
        const int nofOutputValues = 1; // value head has 1 output
        const int nofFilters = 12; //64- the convolution layer has 64 filters
        const int nofConvLayers = 11; // 13- currently 13 conv layers, 1 input, 2 in each of 6 residual layers
        const int nofResidualLayers = 5; // 6- half of (conv-1), 1 conv layer is for input (heads are seperate)
        const int nofPolicyPlanes = 12; // 32- for some reason we only want 32 planes in policy/value heads (the input to is 64 and
        const int nofValuePlanes = 12; //32- conv makes it 32) [cheat sheet alphazero go -> 2]
        const float softmaxTemperature = 1.0f;

        // for input layer
        public float[] input = new float[nofPlanes * gameboardHeight * gameboardWidth]; // input to complete NN
        public float[] outputConvFilter = new float[nofFilters * gameboardHeight * gameboardWidth];
        public float[] firstConvFilterWeights = new float[nofFilters* nofPlanes * filterHeight * filterWidth]; // weights

        // for residual tower
        public float[] inputResidualLayer = new float[nofFilters * gameboardHeight * gameboardWidth]; // input to residual layer
        public float[] outputResidualLayer = new float[nofFilters * gameboardHeight * gameboardWidth];
        public float[] temporary = new float[nofFilters * gameboardHeight * gameboardWidth];
        public float[] convFilterWeights = new float[(nofConvLayers-1) * nofFilters * nofFilters * filterHeight * filterWidth]; // weights

        // for policy layer
        public float[] convWeightsPolicy = new float[nofPolicyPlanes* nofFilters]; // weights 1x1xnofplanes filters
        public float[] BNMeansPolicy = new float[nofPolicyPlanes]; // weights
        public float[] BNStddevPolicy = new float[nofPolicyPlanes]; // weights
        public float[] BNBetaPolicy = new float[nofPolicyPlanes];
        public float[] BNGammaPolicy = new float[nofPolicyPlanes];
        public float[] policyConnectionWeights = new float[gameboardHeight * gameboardWidth* nofPolicyPlanes * nofOutputPolicies]; // weights
        public float[] policyBiases = new float[nofOutputPolicies]; // weights
        public float[] inputFCLayerPolicy = new float[nofPolicyPlanes * gameboardHeight* gameboardWidth];
        public float[] outputPolicyData = new float[nofOutputPolicies];

        // for value layer
        public float[] convWeightsValue1 = new float[nofFilters* nofValuePlanes]; // 1x1 filters, 32 of them for 64 input planes // weights
        public float[] convWeightsValue2 = new float[128]; // weights
        public float[] BNMeansValue = new float[nofValuePlanes]; // weights
        public float[] BNStddevValue = new float[nofValuePlanes]; // weights
        public float[] BNBetaValue = new float[nofValuePlanes];
        public float[] BNGammaValue = new float[nofValuePlanes];
        public float[] valueConnectionWeights = new float[gameboardHeight * gameboardWidth*nofValuePlanes * 128]; // weights
        public float[] valueBiases = new float[128]; // weights
        public float[] valueBiasLast = new float[1]; // weights
        public float[] inputFCLayerValue = new float[nofValuePlanes *gameboardHeight*gameboardWidth];
        public float[] outputValueData = new float[nofValuePlanes * gameboardHeight * gameboardWidth];
        public float[] temporaryValueData = new float[128];

        // for all layers
        public float[] BNMeans = new float[nofConvLayers * nofFilters]; // UNTRAINABLE
        public float[] BNStddev = new float[nofConvLayers * nofFilters]; // UNTRAINABLE

        public float[] BNBetas = new float[nofConvLayers*nofFilters];
        public float[] BNGammas = new float[nofConvLayers* nofFilters];

        // output of NN
        public float[] softmaxPolicy = new float[nofOutputPolicies];
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
            InitializeWeights();
        }
       
        public NeuralNetwork(String file)
        {
            ReadWeightsFromFile(file);
        }
        public void OpenCLInit(int aGlobalID)
        {
            globalID = aGlobalID;
            writer = OpenCL.InputChannel.Writer;
            reader = OpenCL.ResponseChannels[globalID].Reader;
            OpenCL.EnqueueWeights(this);
        }
        public NeuralNetwork(List<float> aWeights, List<float> aUntrainableWeights)
        {
            weights = aWeights;
            untrainable_weights = aUntrainableWeights;
            ParseWeights();
        }
        public Tuple<float[], float> Predict(TicTacToePosition pos)
        {
            // returns array of move evals and V
            /*Not using game history, not using caching*/
            int[] tmp = new int[pos.gameBoard.GetLength(0) * pos.gameBoard.GetLength(1)];
            Buffer.BlockCopy(pos.gameBoard, 0, tmp, 0, tmp.Length * sizeof(int));
            List<int> gameBoard = new List<int>(tmp);

            // set nn input
            for (int i = 0; i < Params.boardSizeX * Params.boardSizeY; ++i)
            {   // the board itself
                input[i] = gameBoard[i];
            }

            for (int i = 0; i < Params.boardSizeX*Params.boardSizeY; ++i)
            {   // whose turn it is
                input[Params.boardSizeX * Params.boardSizeY + i] = pos.sideToMove == Player.X ? 1 : -1;
            }
            if (GPU_PREDICT)
            {
                Job job = new Job();
                job.input = input.ToList();
                job.globalID = globalID;
                writer.TryWrite(job);

                Task t = Consume(reader);
                t.Wait();

                return Tuple.Create(softmaxPolicy, winrateSigOut);
            }
            else
            {
                return ForwardPassCPU(input);
            }
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
        public void EnqueueWeights()
        {
            OpenCL.EnqueueWeights(this);
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
                for (int k = 0; k < Params.boardSizeX * Params.boardSizeY; ++k)
                {   // the board itself
                    input[k] = RandomNr.GetInt(-1,2);
                }

                int sideToMove = RandomNr.GetInt(0, 2) * 2 -1;
                for (int k = 0; k < Params.boardSizeX * Params.boardSizeY; ++k)
                {   // whose turn it is
                    input[Params.boardSizeX * Params.boardSizeY + k] = sideToMove;
                }

                /*Conv layer */
                Convolution(input, outputConvFilter, firstConvFilterWeights, nofPlanes, nofFilters, filterWidth, filterHeight, 0);

                /* copy to intermediate */
                intermediateData[i] = new float[outputConvFilter.Length];
                outputConvFilter.CopyTo(intermediateData[i], 0);
            }
            /* calculate the means and stddev..*/
            CalculateBNMeansAndStddev(intermediateData, BNMeans, BNStddev, BATCHSIZE, nofFilters, 0);

            // ..apply means and stddev..
            for (int i = 0; i < BATCHSIZE; ++i)
            {
                BN(intermediateData[i], inputResidualLayer, BNMeans, BNStddev, nofFilters, 0, BNGammas, BNBetas);
                intermediateData[i] = new float[inputResidualLayer.Length];
                inputResidualLayer.CopyTo(intermediateData[i], 0);
            }

            // ---------------------------------- residual tower -------------------------------------------
            // residual tower
            for (int index = 0; index < nofResidualLayers; index += 1)
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
                    Convolution(intermediateData[i], outputResidualLayer, convFilterWeights, nofFilters, nofFilters, filterWidth, filterHeight, index * 2);

                    /* copy to intermediate */
                    intermediateData[i] = new float[outputResidualLayer.Length];
                    outputResidualLayer.CopyTo(intermediateData[i], 0);
                }
                CalculateBNMeansAndStddev(intermediateData, BNMeans, BNStddev, BATCHSIZE, nofFilters, index * 2+1);

                // apply the means and stddev
                for (int i = 0; i < BATCHSIZE; ++i)
                {
                    BN(intermediateData[i], outputResidualLayer, BNMeans, BNStddev, nofFilters, index * 2+1, BNGammas, BNBetas);

                    /* copy to intermediate */
                    intermediateData[i] = new float[outputResidualLayer.Length];
                    outputResidualLayer.CopyTo(intermediateData[i], 0);
                }

                for (int i = 0; i < BATCHSIZE; ++i)
                {
                    Convolution(intermediateData[i], temporary, convFilterWeights, nofFilters, nofFilters, filterWidth, filterHeight, index * 2 + 1);
                    AddResidual(temporary, temporary, residualSave[i]);

                    /* copy to intermediate */
                    intermediateData[i] = new float[temporary.Length];
                    temporary.CopyTo(intermediateData[i], 0);
                }
                CalculateBNMeansAndStddev(intermediateData, BNMeans, BNStddev, BATCHSIZE, nofFilters, index * 2+2);

                // apply the means and stddev 
                for (int i = 0; i < BATCHSIZE; ++i)
                {
                    BN(intermediateData[i], outputResidualLayer, BNMeans, BNStddev, nofFilters, index * 2 +2, BNGammas, BNBetas);

                    /* copy to intermediate */
                    intermediateData[i] = new float[outputResidualLayer.Length];
                    outputResidualLayer.CopyTo(intermediateData[i], 0);
                }
            }

            // ---------------------------------- value head -------------------------------------------
            float[][] intermediateData2 = new float[BATCHSIZE][];
            for (int i = 0; i < BATCHSIZE; ++i)
            {
                Convolution(intermediateData[i], outputValueData, convWeightsValue1, nofFilters, nofValuePlanes, 1, 1, 0);

                /* copy to intermediate */
                intermediateData2[i] = new float[outputValueData.Length];
                outputValueData.CopyTo(intermediateData2[i], 0);
            }
            CalculateBNMeansAndStddev(intermediateData2, BNMeansValue, BNStddevValue, BATCHSIZE, nofValuePlanes, 0);

            // ---------------------------------- policy head -------------------------------------------
            for (int i = 0; i < BATCHSIZE; ++i)
            {
                Convolution(intermediateData[i], inputFCLayerPolicy, convWeightsPolicy, nofFilters, nofPolicyPlanes, 1, 1, 0);

                /* copy to intermediate */
                intermediateData2[i] = new float[inputFCLayerPolicy.Length];
                inputFCLayerPolicy.CopyTo(intermediateData2[i], 0);
            }
            CalculateBNMeansAndStddev(intermediateData2, BNMeansPolicy, BNStddevPolicy, BATCHSIZE, nofPolicyPlanes, 0);
        }
        private void CalculateBNMeansAndStddev(float[][] intermediateData, float[] BN_means, float[] BN_stddev, int BATCHSIZE, int nofFilters, int index)
        {
            // calc BN means
            for (int i = 0; i < BATCHSIZE; ++i)
            {
                for (int filter = 0; filter < nofFilters; ++filter)
                {
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
            }
        }
        public Tuple<float[], float> ForwardPassCPU(float[] input)
        {
            /*Conv layer */
            Convolution(input, outputConvFilter, firstConvFilterWeights, nofPlanes, nofFilters, filterWidth, filterHeight, 0);
            BN(outputConvFilter, inputResidualLayer, BNMeans, BNStddev, nofFilters, 0, BNGammas, BNBetas);

            /*Residual tower*/
            for (int index = 0; index < nofResidualLayers; index += 1) {
                Convolution(inputResidualLayer, outputResidualLayer, convFilterWeights, nofFilters, nofFilters, filterWidth, filterHeight, index*2);
                BN(outputResidualLayer, outputResidualLayer, BNMeans, BNStddev, nofFilters, index*2+1, BNGammas, BNBetas);
                Convolution(outputResidualLayer, temporary, convFilterWeights, nofFilters, nofFilters, filterWidth, filterHeight, index*2+1);
                BNWithResidual(temporary, outputResidualLayer, inputResidualLayer, BNMeans, BNStddev, nofFilters, index*2+2, BNGammas, BNBetas);
                
                // temporary holds result
                Array.Copy(outputResidualLayer, 0, inputResidualLayer, 0, outputResidualLayer.Length);
            }

            /*value head*/
            Convolution(inputResidualLayer, outputValueData, convWeightsValue1, nofFilters, nofValuePlanes, 1, 1, 0);
            BN(outputValueData, outputValueData, BNMeansValue, BNStddevValue, nofValuePlanes, 0, BNGammaValue, BNBetaValue);
            FCLayer(outputValueData, temporaryValueData, valueConnectionWeights, valueBiases,  true); // with rectifier
            FCLayer(temporaryValueData, winrateOut, convWeightsValue2, valueBiasLast, false); // 1 output, 1 bias
            winrateSigOut = (1.0f + (float)Math.Tanh(winrateOut[0])) / 2.0f;

            /*policy head*/
            Convolution(inputResidualLayer, inputFCLayerPolicy, convWeightsPolicy, nofFilters, nofPolicyPlanes, 1, 1, 0);
            BN(inputFCLayerPolicy, inputFCLayerPolicy, BNMeansPolicy, BNStddevPolicy, nofPolicyPlanes, 0, BNGammaPolicy, BNBetaPolicy);
            FCLayer(inputFCLayerPolicy, outputPolicyData, policyConnectionWeights, policyBiases, false); // without rectifier
            Softmax(outputPolicyData, softmaxPolicy, softmaxTemperature);

            return Tuple.Create(softmaxPolicy, winrateSigOut);
        }
        public void ApplyWeightDecay()
        {
            for(int i = 0; i < weights.Count; ++i)
            {
                weights[i] *= Params.weightDecayFactor;
            }
        }
        private void Convolution(float[] input, float[] output, float[] convWeights,
            int nofInputPlanes, int nofFilters, int filterWidth, int filterHeight, int index)
        {
            // convolution on gameboard_width*gameboardHeight*depth
            // with nofFilters filters of filterWidth*filterHeight*nofInputPlanes size
            // resulting in gameboard_width*gameboardHeight*x volume
            // zero padding

            for (int u = 0; u < output.Length; ++u)
            {
                output[u] = 0.0f;
            }
            for (int i = 0; i < nofFilters; ++i) { 
                // apply each of the filters to the complete input..
                for (int j = 0; j < nofInputPlanes; ++j)
                {
                    for (int k = 0; k < gameboardHeight; ++k)
                    {
                        for (int l = 0; l < gameboardWidth; ++l)
                        {
                            // looking at a 1x1x1 of the input here, we sum up the 3x3 neighbors (depending on filter size)
                            for (int x = 0; x < filterHeight; ++x)
                            {
                                for (int y = 0; y < filterWidth; ++y)
                                {
                                    // going through the neighbors
                                    if (k - filterHeight / 2 + x < 0 || k - filterHeight / 2 + x >= gameboardHeight ||
                                        l - filterWidth / 2 + y < 0 || l - filterWidth / 2 + y >= gameboardWidth)
                                    {
                                        // the filter is out of bounds, set to 0 (0 padding)
                                        continue;
                                    }
                                    output[i * gameboardHeight * gameboardWidth + k * gameboardWidth + l] += 
                                        input[j * gameboardHeight * gameboardWidth + k * gameboardWidth + l] *
                                        convWeights[
                                            index * nofFilters * nofInputPlanes * filterHeight * filterWidth +
                                            i * nofInputPlanes * filterHeight * filterWidth + 
                                            j * filterHeight * filterWidth +
                                            x * filterWidth + y];
                                }
                            }
                            // add the bias in BN to the means
                        }
                    }
                }
            }
           // });
            // after summing all values, divide by number of summed up fields
            for (int u = 0; u < output.Length; ++u)
            {
                output[u] /= nofInputPlanes * filterHeight * filterWidth;
            }
        }
        private void BN(float[] input, float[] output, float[] BNMeans, float[] BNStdDev, int nofFilters, int index, float[] BNGammas, float[] BNBetas)
        {
            // without residual add
            for (int i = 0; i < nofFilters; ++i)
            {
                // go through each plane coming into BN and apply to each element the means and stddev..
                for (int j = 0; j < input.Length/nofFilters; ++j)
                {
                    // we know the size of one plane by dividing input through number of plans (input.length/noffilters)
                    // batch norm/ batch stddev
                    /* see Alg 1: https://arxiv.org/pdf/1502.03167.pdf */
                    float x_til = (float)((input[i * input.Length / nofFilters + j] - BNMeans[index * nofFilters + i])/
                        (Math.Sqrt(BNStdDev[index * nofFilters + i]+0.0001f)));
                    output[i * input.Length / nofFilters + j] = BNGammas[index * nofFilters + i] *x_til+BNBetas[index * nofFilters + i];

                    // relu
                    if (output[i * input.Length / nofFilters + j] < 0.0f)
                        output[i* input.Length / nofFilters + j] = 0.0f;
                }
            }
        }
        private void BNWithResidual(float[] input, float[] output, float[] residual, float[] BNMeans, float[] BNStdDev, int nofFilters, int index, float[] BNGammas, float[] BNBetas)
        {
            for (int i = 0; i < nofFilters; ++i)
            {
                for (int j = 0; j < gameboardWidth * gameboardHeight; ++j)
                {
                    // batch norm/ batch stddev
                    float x_til = (float)((input[i * input.Length / nofFilters + j] + 
                        residual[i * input.Length / nofFilters + j] - BNMeans[index * nofFilters + i]) /
                        (Math.Sqrt(BNStdDev[index * nofFilters + i] + 0.0001f)));

                    output[i * input.Length / nofFilters + j] = BNGammas[index * nofFilters + i] * x_til + BNBetas[index * nofFilters + i];

                    // relu
                    if (output[i * input.Length / nofFilters + j] < 0.0f)
                        output[i * input.Length / nofFilters + j] = 0.0f;
                }
            }
        }
        private void AddResidual(float[] input, float[] output, float[] residual)
        {
            for (int i = 0; i < nofFilters; ++i)
            {
                for (int j = 0; j < gameboardWidth * gameboardHeight; ++j)
                {
                    output[i * gameboardWidth * gameboardHeight + j] = input[i * gameboardWidth * gameboardHeight + j] + residual[i * gameboardWidth * gameboardHeight + j];
                }
            }
        }
        private void FCLayer(float[] input, float[] output, float[] connectionWeights, float[] outputBiases, bool rectifier)
        {
            for (int u = 0; u < output.Length; ++u)
            {
                output[u] = 0.0f;
            }
            for (int i = 0; i < output.Length; ++i)
            {
                for (int j = 0; j < input.Length; ++j)
                {
                    output[i] += input[j] * connectionWeights[i * input.Length + j];
                }
                output[i] /= input.Length;
                output[i] += outputBiases[i];

                if(rectifier && output[i] < 0.0f)
                {
                    output[i] = 0.0f;
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
                    data[i] = 0.0f;
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
            for (int i = 0; i < convWeightsValue2.Length; ++i)
            {
                convWeightsValue2[i] = weights[count];
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

            for (int i = 0; i < convWeightsValue2.Length-1; ++i)
            {
                fileWriter.Write(convWeightsValue2[i] + " ");
            }
            fileWriter.Write(convWeightsValue2[convWeightsValue2.Length - 1] + "\n");

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
        public void InitializeWeights()
        {
            for (int i = 0; i < firstConvFilterWeights.Length; ++i)
            {
                firstConvFilterWeights[i] = RandomNr.GetGaussianFloat();
            }
            for (int i = 0; i < convFilterWeights.Length; ++i)
            {
                convFilterWeights[i] = RandomNr.GetGaussianFloat();
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
                convWeightsPolicy[i] = RandomNr.GetGaussianFloat();
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
                policyConnectionWeights[i] = RandomNr.GetGaussianFloat();
            }
            for (int i = 0; i < policyBiases.Length; ++i)
            {
                policyBiases[i] = RandomNr.GetGaussianFloat();
            }
            for (int i = 0; i < convWeightsValue1.Length; ++i)
            {
                convWeightsValue1[i] = RandomNr.GetGaussianFloat();
            }
            for (int i = 0; i < convWeightsValue2.Length; ++i)
            {
                convWeightsValue2[i] = RandomNr.GetGaussianFloat();
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
                valueConnectionWeights[i] = RandomNr.GetGaussianFloat();
            }
            for (int i = 0; i < valueBiases.Length; ++i)
            {
                valueBiases[i] = RandomNr.GetGaussianFloat();
            }
            for (int i = 0; i < valueBiasLast.Length; ++i)
            {
                valueBiasLast[i] = RandomNr.GetGaussianFloat();
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
 
 