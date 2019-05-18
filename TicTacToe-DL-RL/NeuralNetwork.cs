using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    class NeuralNetwork
    {
        // params
        const int width = 5;
        const int height = 5;
        const int filterWidth = 3;
        const int filterHeight = 3;
        const int nofPlanes = 2; // = input channels, 1 plane is board 3x3 + 1 plane color 3x3
        const int nofOutputPolicies = 25; // policy net has 9 outputs (1 per potential move)
        const int nofOutputValues = 1; // value head has 1 output
        const int nofFilters = 6; //64- the convolution layer has 64 filters
        const int nofConvLayers = 9; // 13- currently 13 conv layers, 1 input, 2 in each of 6 residual layers
        const int nofResidualLayers = 4; // 6- half of (conv-1), 1 conv layer is for input (heads are seperate)
        const int nofPolicyPlanes = 4; // 32- for some reason we only want 32 planes in policy/value heads (the input to is 64 and
        const int nofValuePlanes = 4; //32- conv makes it 32) [cheat sheet alphazero go -> 2]
        const float softmaxTemperature = 1.0f;

        // for input layer
        float[] input = new float[nofPlanes * height * width]; // input to complete NN
        float[] outputConvFilter = new float[nofFilters * height * width];
        float[] firstConvFilterWeights = new float[nofFilters* nofPlanes * filterHeight * filterWidth]; // weights

        // for residual tower
        float[] inputResidualLayer = new float[nofFilters * height * width]; // input to residual layer
        float[] outputResidualLayer = new float[nofFilters * height * width];
        float[] temporary = new float[nofFilters * height * width];
        float[] convFilterWeights = new float[(nofConvLayers-1) * nofFilters * nofFilters * filterHeight * filterWidth]; // weights

        // for policy layer
        float[] convolutionWeightsPolicy = new float[nofPolicyPlanes* nofFilters]; // weights 1x1xnofplanes filters
        float[] convolutionBiasesPolicy = new float[nofPolicyPlanes* nofFilters]; // weights
        float[] batchnormMeansPolicy = new float[nofPolicyPlanes]; // weights
        float[] batchnormStddevPolicy = new float[nofPolicyPlanes]; // weights
        float[] batchnormBetaPolicy = new float[nofPolicyPlanes];
        float[] batchnormGammaPolicy = new float[nofPolicyPlanes];
        float[] policyConnectionWeights = new float[height * width* nofPolicyPlanes * nofOutputPolicies]; // weights
        float[] policyBiases = new float[nofOutputPolicies]; // weights
        float[] inputFullyConnectedLayerPolicy = new float[nofPolicyPlanes * height* width];
        float[] outputPolicyData = new float[nofOutputPolicies];

        // for value layer
        float[] convolutionWeightsValue1 = new float[nofFilters* nofValuePlanes]; // 1x1 filters, 32 of them for 64 input planes // weights
        float[] convolutionWeightsValue2 = new float[128]; // weights
        float[] batchnormMeansValue = new float[nofValuePlanes]; // weights
        float[] batchnormStddevValue = new float[nofValuePlanes]; // weights
        float[] batchnormBetaValue = new float[nofValuePlanes];
        float[] batchnormGammaValue = new float[nofValuePlanes];
        float[] valueConnectionWeights = new float[height * width*nofValuePlanes * 128]; // weights
        float[] valueBiases = new float[128]; // weights
        float[] valueBiasLast = new float[1]; // weights
        float[] inputFullyConnectedLayerValue = new float[nofValuePlanes *height*width];
        float[] outputValueData = new float[nofValuePlanes * height * width];
        float[] temporaryValueData = new float[128];

        // for all layers
        float[] convBiases = new float[nofConvLayers * nofFilters]; // weights
        float[] batchnorm_means = new float[nofConvLayers * nofFilters]; // UNTRAINABLE
        float[] batchnorm_stddev = new float[nofConvLayers * nofFilters]; // UNTRAINABLE

        float[] batchnorm_betas = new float[nofConvLayers*nofFilters];
        float[] batchnorm_gammas = new float[nofConvLayers* nofFilters];

        // output of NN
        float[] softmaxPolicy = new float[nofOutputPolicies];
        float[] winrateOut = new float[1];

        // complete weights
        public List<float> weights = new List<float>();
        public List<float> untrainable_weights = new List<float>();

        public NeuralNetwork()
        {
            InitializeWeights();
            CalculateVirtualBatchnorms();
            //SaveWeightsToFile("weights_best.txt");
            //ReadWeightsFromFile("weights.txt");
            //ReadWeightsFromFile("weights.txt");
            //for (int i = 0; i < convBiases.Length; ++i) {
            //    batchnorm_means[i] -= convBiases[i];
            //    convBiases[i] = 0.0f;
            //}
        }
        public NeuralNetwork(String file)
        {
            ReadWeightsFromFile(file);
        }
        public NeuralNetwork(List<float> aWeights, List<float> aUntrainableWeights)
        {
            weights = aWeights;
            untrainable_weights = aUntrainableWeights;
            ParseWeights();
        }
        public Tuple<float[], float> Predict(TicTacToePosition pos)
        {
            // temp
            //return Tuple.Create(softmaxPolicy, 0.0f);
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
            return ForwardPassCPU(input);
        }
        private void CalculateVirtualBatchnorms()
        {
            int BATCHSIZE = 100;
            float[][] intermediateData = new float[BATCHSIZE][];

            Array.Clear(batchnorm_means, 0, batchnorm_means.Length);
            Array.Clear(batchnorm_stddev, 0, batchnorm_stddev.Length);
            Array.Clear(batchnormMeansPolicy, 0, batchnormMeansPolicy.Length);
            Array.Clear(batchnormStddevPolicy, 0, batchnormStddevPolicy.Length);
            Array.Clear(batchnormMeansValue, 0, batchnormMeansValue.Length);
            Array.Clear(batchnormStddevValue, 0, batchnormStddevValue.Length);

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
            CalculateBatchNormMeansAndStddev(intermediateData, batchnorm_means, batchnorm_stddev, BATCHSIZE, nofFilters, 0);

            // ..apply means and stddev..
            for (int i = 0; i < BATCHSIZE; ++i)
            {
                BatchNorm(intermediateData[i], inputResidualLayer, batchnorm_means, batchnorm_stddev, nofFilters, 0);
                intermediateData[i] = new float[inputResidualLayer.Length];
                inputResidualLayer.CopyTo(intermediateData[i], 0);
            }

            // ---------------------------------- residual tower -------------------------------------------
            // residual tower
            for (int index = 0; index < nofResidualLayers; index += 1)
            {
                for (int i = 0; i < BATCHSIZE; ++i)
                {
                    // .. apply the means and stddev..
                    Convolution(intermediateData[i], outputResidualLayer, convFilterWeights, nofFilters, nofFilters, filterWidth, filterHeight, index * 2);

                    /* copy to intermediate */
                    intermediateData[i] = new float[outputResidualLayer.Length];
                    outputResidualLayer.CopyTo(intermediateData[i], 0);
                }
                CalculateBatchNormMeansAndStddev(intermediateData, batchnorm_means, batchnorm_stddev, BATCHSIZE, nofFilters, index * 2+1);

                // apply the means and stddev
                for (int i = 0; i < BATCHSIZE; ++i)
                {
                    BatchNorm(intermediateData[i], outputResidualLayer, batchnorm_means, batchnorm_stddev, nofFilters, index * 2+1);

                    /* copy to intermediate */
                    intermediateData[i] = new float[outputResidualLayer.Length];
                    outputResidualLayer.CopyTo(intermediateData[i], 0);
                }

                for (int i = 0; i < BATCHSIZE; ++i)
                {
                    Convolution(intermediateData[i], temporary, convFilterWeights, nofFilters, nofFilters, filterWidth, filterHeight, index * 2 + 1);
                    AddResidual(temporary, temporary, inputResidualLayer);

                    /* copy to intermediate */
                    intermediateData[i] = new float[temporary.Length];
                    temporary.CopyTo(intermediateData[i], 0);
                }
                CalculateBatchNormMeansAndStddev(intermediateData, batchnorm_means, batchnorm_stddev, BATCHSIZE, nofFilters, index * 2+2);

                // apply the means and stddev 
                for (int i = 0; i < BATCHSIZE; ++i)
                {
                    BatchNorm(intermediateData[i], outputResidualLayer, batchnorm_means, batchnorm_stddev, nofFilters, index * 2 +2);

                    /* copy to intermediate */
                    intermediateData[i] = new float[outputResidualLayer.Length];
                    outputResidualLayer.CopyTo(intermediateData[i], 0);
                }
            }

            // ---------------------------------- value head -------------------------------------------
            float[][] intermediateData2 = new float[BATCHSIZE][];
            for (int i = 0; i < BATCHSIZE; ++i)
            {
                Convolution(intermediateData[i], outputValueData, convolutionWeightsValue1, nofFilters, nofValuePlanes, 1, 1, 0);

                /* copy to intermediate */
                intermediateData2[i] = new float[outputValueData.Length];
                outputValueData.CopyTo(intermediateData2[i], 0);
            }
            CalculateBatchNormMeansAndStddev(intermediateData2, batchnormMeansValue, batchnormStddevValue, BATCHSIZE, nofValuePlanes, 0);

            // ---------------------------------- policy head -------------------------------------------
            for (int i = 0; i < BATCHSIZE; ++i)
            {
                Convolution(intermediateData[i], inputFullyConnectedLayerPolicy, convolutionWeightsPolicy, nofFilters, nofPolicyPlanes, 1, 1, 0);

                /* copy to intermediate */
                intermediateData2[i] = new float[inputFullyConnectedLayerPolicy.Length];
                inputFullyConnectedLayerPolicy.CopyTo(intermediateData2[i], 0);
            }
            CalculateBatchNormMeansAndStddev(intermediateData2, batchnormMeansPolicy, batchnormStddevPolicy, BATCHSIZE, nofPolicyPlanes, 0);
        }
        private void CalculateBatchNormMeansAndStddev(float[][] intermediateData, float[] batchnorm_means, float[] batchnorm_stddev, int BATCHSIZE, int nofFilters, int index)
        {
            // calc batchnorm means
            for (int i = 0; i < BATCHSIZE; ++i)
            {
                for (int filter = 0; filter < nofFilters; ++filter)
                {
                    for (int k = intermediateData[i].Length/nofFilters * filter; k < intermediateData[i].Length / nofFilters * (filter+1); k++) 
                    {
                        // read out correct plane,all planes are sequential in intermediatedata
                        batchnorm_means[index * nofFilters + filter] += intermediateData[i][k];
                    }
                }
            }
            for (int filter = 0; filter < nofFilters; ++filter)
            {
                batchnorm_means[index * nofFilters + filter] /= BATCHSIZE* intermediateData[0].Length / nofFilters;
            }

            // calc batchnorm stddev
            for (int i = 0; i < BATCHSIZE; ++i)
            {
                for (int filter = 0; filter < nofFilters; ++filter)
                {
                    for (int k = intermediateData[i].Length / nofFilters*filter; k < intermediateData[i].Length / nofFilters * (filter + 1); k++)
                    {
                        batchnorm_stddev[index * nofFilters + filter] += (float)Math.Pow(intermediateData[i][k] - batchnorm_means[index * nofFilters + filter], 2.0);
                    }
                }
            }
            for (int filter = 0; filter < nofFilters; ++filter)
            {
                batchnorm_stddev[index * nofFilters + filter] /= BATCHSIZE * intermediateData[0].Length / nofFilters;
            }
        }
        public Tuple<float[], float> ForwardPassCPU(float[] input)
        {
            /*Conv layer */
            Convolution(input, outputConvFilter, firstConvFilterWeights, nofPlanes, nofFilters, filterWidth, filterHeight, 0);
            BatchNorm(outputConvFilter, inputResidualLayer, batchnorm_means, batchnorm_stddev, nofFilters, 0);

            /*Residual tower*/
            for (int index = 0; index < nofResidualLayers; index += 1) {
                Convolution(inputResidualLayer, outputResidualLayer, convFilterWeights, nofFilters, nofFilters, filterWidth, filterHeight, index*2);
                BatchNorm(outputResidualLayer, outputResidualLayer, batchnorm_means, batchnorm_stddev, nofFilters, index*2+1);
                Convolution(outputResidualLayer, temporary, convFilterWeights, nofFilters, nofFilters, filterWidth, filterHeight, index*2+1);
                BatchNormWithResidual(temporary, outputResidualLayer, inputResidualLayer, batchnorm_means, batchnorm_stddev, nofFilters, index*2+2);
                
                // temporary holds result
                float[] swap = new float[inputResidualLayer.Length];
                Array.Copy(inputResidualLayer, 0, swap, 0, inputResidualLayer.Length);
                Array.Copy(outputResidualLayer, 0, inputResidualLayer, 0, outputResidualLayer.Length);
                Array.Copy(swap, 0, outputResidualLayer, 0, swap.Length); // TODO: swap? why?
            }

            /*value head*/
            Convolution(inputResidualLayer, outputValueData, convolutionWeightsValue1, nofFilters, nofValuePlanes, 1, 1, 0);
            BatchNorm(outputValueData, outputValueData, batchnormMeansValue, batchnormStddevValue, nofValuePlanes, 0);
            FullyConnectedLayer(outputValueData, temporaryValueData, valueConnectionWeights, valueBiases,  true); // with rectifier
            FullyConnectedLayer(temporaryValueData, winrateOut, convolutionWeightsValue2, valueBiasLast, false); // 1 output, 1 bias
            float winrateSig = (1.0f + (float)Math.Tanh(winrateOut[0])) / 2.0f;

            /*policy head*/
            Convolution(inputResidualLayer, inputFullyConnectedLayerPolicy, convolutionWeightsPolicy, nofFilters, nofPolicyPlanes, 1, 1, 0);
            BatchNorm(inputFullyConnectedLayerPolicy, inputFullyConnectedLayerPolicy, batchnormMeansPolicy, batchnormStddevPolicy, nofPolicyPlanes, 0);
            FullyConnectedLayer(inputFullyConnectedLayerPolicy, outputPolicyData, policyConnectionWeights, policyBiases, false); // without rectifier
            Softmax(outputPolicyData, softmaxPolicy, softmaxTemperature);

            return Tuple.Create(softmaxPolicy, winrateSig);
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
            // convolution on width*height*depth
            // with nofFilters filters of filterWidth*filterHeight*nofInputPlanes size
            // resulting in width*height*x volume
            // zero padding

            for (int u = 0; u < output.Length; ++u)
            {
                output[u] = 0.0f;
            }
            for (int i = 0; i < nofFilters; ++i) { 
                // apply each of the filters to the complete input..
                for (int j = 0; j < nofInputPlanes; ++j)
                {
                    for (int k = 0; k < height; ++k)
                    {
                        for (int l = 0; l < width; ++l)
                        {
                            // looking at a 1x1x1 of the input here, we sum up the 3x3 neighbors (depending on filter size)
                            for (int x = 0; x < filterHeight; ++x)
                            {
                                for (int y = 0; y < filterWidth; ++y)
                                {
                                    // going through the neighbors
                                    if (k - filterHeight / 2 + x < 0 || k - filterHeight / 2 + x >= height ||
                                        l - filterWidth / 2 + y < 0 || l - filterWidth / 2 + y >= width)
                                    {
                                        // the filter is out of bounds, set to 0 (0 padding)
                                        continue;
                                    }
                                    output[i * height * width + k * width + l] += 
                                        input[j * height * width + k * width + l] *
                                        convWeights[
                                            index * nofFilters * nofInputPlanes * filterHeight * filterWidth +
                                            i * nofInputPlanes * filterHeight * filterWidth + 
                                            j * filterHeight * filterWidth +
                                            x * filterWidth + y];
                                }
                            }
                            // add the bias in batchnorm to the means
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
        private void BatchNorm(float[] input, float[] output, float[] batchNormMeans, float[] batchNormStdDev, int nofFilters, int index)
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
                    float x_til = (float)((input[i * input.Length / nofFilters + j] - batchNormMeans[index * nofFilters + i])/
                        (Math.Sqrt(batchNormStdDev[index * nofFilters + i]+0.0001f)));
                    output[i * input.Length / nofFilters + j] = batchnorm_gammas[index * nofFilters + i] *x_til+batchnorm_betas[index * nofFilters + i];

                    // relu
                    if (output[i * input.Length / nofFilters + j] < 0.0f)
                        output[i* input.Length / nofFilters + j] = 0.0f;
                }
            }
        }
        private void BatchNormWithResidual(float[] input, float[] output, float[] residual, float[] batchNormMeans, float[] batchNormStdDev, int nofFilters, int index)
        {
            for (int i = 0; i < nofFilters; ++i)
            {
                for (int j = 0; j < width * height; ++j)
                {
                    // batch norm/ batch stddev
                    float x_til = (float)((input[i * input.Length / nofFilters + j] + residual[i * input.Length / nofFilters + j] - batchNormMeans[index * nofFilters + i]) /
    (Math.Sqrt(batchNormStdDev[index * nofFilters + i] + 0.0001f)));

                    output[i * input.Length / nofFilters + j] = batchnorm_gammas[index * nofFilters + i] * x_til + batchnorm_betas[index * nofFilters + i];

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
                for (int j = 0; j < width * height; ++j)
                {
                    output[i * width * height + j] = input[i * width * height + j] + residual[i * width * height + j];
                }
            }
        }
        private void FullyConnectedLayer(float[] input, float[] output, float[] connectionWeights, float[] outputBiases, bool rectifier)
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
            for (int i = 0; i < convBiases.Length; ++i)
            {
                convBiases[i] = weights[count];
                count++;
            }
            for (int i = 0; i < batchnorm_means.Length; ++i)
            {
                batchnorm_means[i] = untrainable_weights[untrainable_count];
                untrainable_count++;
            }
            for (int i = 0; i < batchnorm_stddev.Length; ++i)
            {
                batchnorm_stddev[i] = untrainable_weights[untrainable_count];
                untrainable_count++;
            }
            for (int i = 0; i < convolutionWeightsPolicy.Length; ++i)
            {
                convolutionWeightsPolicy[i] = weights[count];
                count++;
            }
            for (int i = 0; i < convolutionBiasesPolicy.Length; ++i)
            {
                convolutionBiasesPolicy[i] = weights[count];
                count++;
            }
            for (int i = 0; i < batchnormMeansPolicy.Length; ++i)
            {
                batchnormMeansPolicy[i] = untrainable_weights[untrainable_count];
                untrainable_count++;
            }
            for (int i = 0; i < batchnormStddevPolicy.Length; ++i)
            {
                batchnormStddevPolicy[i] = untrainable_weights[untrainable_count];
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
            for (int i = 0; i < convolutionWeightsValue1.Length; ++i)
            {
                convolutionWeightsValue1[i] = weights[count];
                count++;
            }         
            for (int i = 0; i < convolutionWeightsValue2.Length; ++i)
            {
                convolutionWeightsValue2[i] = weights[count];
                count++;
            }
            for (int i = 0; i < batchnormMeansValue.Length; ++i)
            {
                batchnormMeansValue[i] = untrainable_weights[untrainable_count];
                untrainable_count++;
            }
            for (int i = 0; i < batchnormStddevValue.Length; ++i)
            {
                batchnormStddevValue[i] = untrainable_weights[untrainable_count];
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
            for (int i = 0; i < batchnormBetaPolicy.Length; ++i)
            {
                batchnormBetaPolicy[i] = weights[count];
                count++;
            }
            for (int i = 0; i < batchnormGammaPolicy.Length; ++i)
            {
                batchnormGammaPolicy[i] = weights[count];
                count++;
            }
            for (int i = 0; i < batchnormBetaValue.Length; ++i)
            {
                batchnormBetaValue[i] = weights[count];
                count++;
            }
            for (int i = 0; i < batchnormGammaValue.Length; ++i)
            {
                batchnormGammaValue[i] = weights[count];
                count++;
            }

            for (int i = 0; i < batchnorm_betas.Length; ++i)
            {
                batchnorm_betas[i] = weights[count];
                count++;
            }
            for (int i = 0; i < batchnorm_gammas.Length; ++i)
            {
                batchnorm_gammas[i] = weights[count];
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

            for (int i = 0; i < convBiases.Length-1; ++i)
            {
                fileWriter.Write(convBiases[i] + " ");
            }
            fileWriter.Write(convBiases[convBiases.Length - 1] + "\n");

            for (int i = 0; i < batchnorm_means.Length-1; ++i)
            {
                fileWriter2.Write(batchnorm_means[i] + " ");
            }
            fileWriter2.Write(batchnorm_means[batchnorm_means.Length - 1] + "\n");

            //w = 1.0f / std::sqrt(w + epsilon) on read of bn stddivs
            for (int i = 0; i < batchnorm_stddev.Length-1; ++i)
            {
                fileWriter2.Write(batchnorm_stddev[i] + " ");
            }
            fileWriter2.Write(batchnorm_stddev[batchnorm_stddev.Length - 1] + "\n");

            // policy head
            for (int i = 0; i < convolutionWeightsPolicy.Length-1; ++i)
            {
                fileWriter.Write(convolutionWeightsPolicy[i] + " ");
            }
            fileWriter.Write(convolutionWeightsPolicy[convolutionWeightsPolicy.Length - 1] + "\n");

            for (int i = 0; i < convolutionBiasesPolicy.Length-1; ++i)
            {
                fileWriter.Write(convolutionBiasesPolicy[i] + " ");
            }
            fileWriter.Write(convolutionBiasesPolicy[convolutionBiasesPolicy.Length - 1] + "\n");

            for (int i = 0; i < batchnormMeansPolicy.Length-1; ++i)
            {
                fileWriter2.Write(batchnormMeansPolicy[i] + " ");
            }
            fileWriter2.Write(batchnormMeansPolicy[batchnormMeansPolicy.Length - 1] + "\n");

            for (int i = 0; i < batchnormStddevPolicy.Length-1; ++i)
            {
                fileWriter2.Write(batchnormStddevPolicy[i] + " ");
            }
            fileWriter2.Write(batchnormStddevPolicy[batchnormStddevPolicy.Length - 1] + "\n");

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
            for (int i = 0; i < convolutionWeightsValue1.Length-1; ++i)
            {
                fileWriter.Write(convolutionWeightsValue1[i] + " ");
            }
            fileWriter.Write(convolutionWeightsValue1[convolutionWeightsValue1.Length - 1] + "\n");

            for (int i = 0; i < convolutionWeightsValue2.Length-1; ++i)
            {
                fileWriter.Write(convolutionWeightsValue2[i] + " ");
            }
            fileWriter.Write(convolutionWeightsValue2[convolutionWeightsValue2.Length - 1] + "\n");

            for (int i = 0; i < batchnormMeansValue.Length-1; ++i)
            {
                fileWriter2.Write(batchnormMeansValue[i] + " ");
            }
            fileWriter2.Write(batchnormMeansValue[batchnormMeansValue.Length - 1] + "\n");

            for (int i = 0; i < batchnormStddevValue.Length-1; ++i)
            {
                fileWriter2.Write(batchnormStddevValue[i] + " ");
            }
            fileWriter2.Write(batchnormStddevValue[batchnormStddevValue.Length - 1] + "\n");

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


            for (int i = 0; i < batchnormBetaPolicy.Length - 1; ++i)
            {
                fileWriter.Write(batchnormBetaPolicy[i] + " ");
            }
            fileWriter.Write(batchnormBetaPolicy[batchnormBetaPolicy.Length - 1] + "\n");
            for (int i = 0; i < batchnormGammaPolicy.Length - 1; ++i)
            {
                fileWriter.Write(batchnormGammaPolicy[i] + " ");
            }
            fileWriter.Write(batchnormGammaPolicy[batchnormGammaPolicy.Length - 1] + "\n");
            for (int i = 0; i < batchnormBetaValue.Length - 1; ++i)
            {
                fileWriter.Write(batchnormBetaValue[i] + " ");
            }
            fileWriter.Write(batchnormBetaValue[batchnormBetaValue.Length - 1] + "\n");
            for (int i = 0; i < batchnormGammaValue.Length - 1; ++i)
            {
                fileWriter.Write(batchnormGammaValue[i] + " ");
            }
            fileWriter.Write(batchnormGammaValue[batchnormGammaValue.Length - 1] + "\n");

            for (int i = 0; i < batchnorm_betas.Length -1; ++i)
            {
                fileWriter.Write(batchnorm_betas[i] + " ");
            }
            fileWriter.Write(batchnorm_betas[batchnorm_betas.Length -1] + "\n");

            for (int i = 0; i < batchnorm_gammas.Length -1; ++i)
            {
                fileWriter.Write(batchnorm_gammas[i] + " ");
            }
            fileWriter.Write(batchnorm_gammas[batchnorm_gammas.Length -1] + "\n");

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
            for (int i = 0; i < convBiases.Length; ++i)
            {
                convBiases[i] = RandomNr.GetGaussianFloat();
            }
            for (int i = 0; i < batchnorm_means.Length; ++i)
            {
                batchnorm_means[i] = 0.0f;
            }
            for (int i = 0; i < batchnorm_stddev.Length; ++i)
            {
                batchnorm_stddev[i] = 1.0f;
            }
            for (int i = 0; i < convolutionWeightsPolicy.Length; ++i)
            {
                convolutionWeightsPolicy[i] = RandomNr.GetGaussianFloat();
            }
            for (int i = 0; i < convolutionBiasesPolicy.Length; ++i)
            {
                convolutionBiasesPolicy[i] = RandomNr.GetGaussianFloat();
            }
            for (int i = 0; i < batchnormMeansPolicy.Length; ++i)
            {
                batchnormMeansPolicy[i] = 0.0f;
            }
            for (int i = 0; i < batchnormStddevPolicy.Length; ++i)
            {
                batchnormStddevPolicy[i] = 1.0f;
            }
            for (int i = 0; i < policyConnectionWeights.Length; ++i)
            {
                policyConnectionWeights[i] = RandomNr.GetGaussianFloat();
            }
            for (int i = 0; i < policyBiases.Length; ++i)
            {
                policyBiases[i] = RandomNr.GetGaussianFloat();
            }
            for (int i = 0; i < convolutionWeightsValue1.Length; ++i)
            {
                convolutionWeightsValue1[i] = RandomNr.GetGaussianFloat();
            }
            for (int i = 0; i < convolutionWeightsValue2.Length; ++i)
            {
                convolutionWeightsValue2[i] = RandomNr.GetGaussianFloat();
            }
            for (int i = 0; i < batchnormMeansValue.Length; ++i)
            {
                batchnormMeansValue[i] = 0.0f;
            }
            for (int i = 0; i < batchnormStddevValue.Length; ++i)
            {
                batchnormStddevValue[i] = 1.0f;
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

            for (int i = 0; i < batchnormBetaPolicy.Length; ++i)
            {
                batchnormBetaPolicy[i] = 0.0f;
            }
            for (int i = 0; i < batchnormGammaPolicy.Length; ++i)
            {
                batchnormGammaPolicy[i] = 1.0f;
            }
            for (int i = 0; i < batchnormBetaValue.Length; ++i)
            {
                batchnormBetaValue[i] = 0.0f;
            }
            for (int i = 0; i < batchnormGammaValue.Length; ++i)
            {
                batchnormGammaValue[i] = 1.0f;
            }

            for (int i = 0; i < batchnorm_betas.Length; ++i)
            {
                batchnorm_betas[i] = 0.0f;
                batchnorm_gammas[i] = 1.0f;
            }
        }
    }
}
 
 