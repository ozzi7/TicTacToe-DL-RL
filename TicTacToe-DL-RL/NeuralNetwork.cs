using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    class NeuralNetwork
    {
        const int width = 3;
        const int height = 3;
        const int filterWidth = 3;
        const int filterHeight = 3;
        const int nofPlanes = 2; // = input channels, 1 plane is board 3x3 + 1 plane color 3x3
        const int nofOutputPolicies = 9; // policy net has 9 outputs (1 per potential move)
        const int nofOutputValues = 1; // value head has 1 output
        const int nofFilters = 64; // the convolution layer has 64 filters
        const int nofConvLayers = 13; // currently 13 conv layers, 1 input, 2 in each of 6 residual layers
        const int nofResidualLayers = 6; // half of (conv-1), 1 conv layer is for input (heads are seperate)
        const int nofPolicyPlanes = 32; // for some reason we only want 32 planes in policy/value heads (the input to is 64 and
        const int nofValuePlanes = 32; // conv makes it 32) [cheat sheet alphazero go -> 2]
        const float softmaxTemperature = 1.0f;

        // for input layer
        float[] input = new float[nofPlanes * height * width]; // input to complete NN
        float[] outputConvFilter = new float[nofFilters * height * width];
        float[] firstConvFilterWeights = new float[nofFilters* nofPlanes * filterHeight * filterWidth];

        // for residual tower
        float[] inputResidualLayer = new float[nofPlanes * height * width]; // input to residual layer
        float[] outputResidualLayer = new float[nofPlanes * height * width];
        float[] temporary = new float[nofPlanes * height * width];
        float[] convFilterWeights = new float[(nofConvLayers-1) * filterHeight * filterWidth];

        // for policy layer
        float[] convolutionWeightsPolicy = new float[nofPolicyPlanes* nofFilters];
        float[] convolutionBiasesPolicy = new float[nofPolicyPlanes* nofFilters];
        float[] batchnormMeansPolicy = new float[nofPolicyPlanes];
        float[] batchnormStddevPolicy = new float[nofPolicyPlanes];
        float[] policyConnectionWeights = new float[width * height * nofPolicyPlanes * nofOutputPolicies];
        float[] policyBiases = new float[nofOutputPolicies];
        float[] inputFullyConnectedLayerPolicy = new float[nofPolicyPlanes * width * height];
        float[] outputPolicyData = new float[nofOutputPolicies];

        // for value layer
        float[] convolutionWeightsValue1 = new float[nofValuePlanes];
        float[] convolutionWeightsValue2 = new float[128];
        float[] batchnormMeansValue = new float[nofValuePlanes];
        float[] batchnormStddevValue = new float[nofValuePlanes];
        float[] valueConnectionWeights = new float[width * height * nofValuePlanes * 128];
        float[] valueBiases = new float[128];
        float[] valueBiasLast = new float[1];
        float[] inputFullyConnectedLayerValue = new float[nofValuePlanes * width * height];
        float[] outputValueData = new float[nofPlanes * width * height];

        // for all
        float[] convBiases = new float[nofConvLayers * nofFilters];
        float[] batchnorm_means = new float[nofConvLayers * nofFilters];
        float[] batchnorm_stddev = new float[nofConvLayers * nofFilters];

        // output of NN
        float[] softmaxPolicy = new float[nofOutputPolicies];
        float[] winrateOut = new float[1];

        public NeuralNetwork()
        {
            for (int i = 0; i < convBiases.Count(); ++i) {
                batchnorm_means[i] -= convBiases[i];
                convBiases[i] = 0.0f;
            }
        }

        public Tuple<List<float>, float> Predict(Position pos)
        {
            /*Not using game history, not using caching*/
            int[] tmp = new int[pos.gameBoard.GetLength(0) * pos.gameBoard.GetLength(1)];
            Buffer.BlockCopy(pos.gameBoard, 0, tmp, 0, tmp.Length * sizeof(int));
            List<int> gameBoard = new List<int>(tmp);

            List<float> inpuData = new List<float>(nofPlanes * width * height);
            List<float> softmaxData = new List<float>(nofOutputPolicies);

            // set nn input
            for (int i = 0; i < 9; ++i)
            {   // whose turn it is
                input[i] = pos.sideToMove;
            }
            for (int i = 0; i < 9; ++i)
            {   // the board itself
                input[9+i] = gameBoard[i];
            }
            ForwardPassCPU(input);
            return null;
        }
        public void ForwardPassCPU(float[] input)
        {
            /*Conv layer */
            Convolution(input, outputConvFilter, firstConvFilterWeights, nofPlanes, nofFilters, filterWidth, filterHeight, 0);
            BatchNorm(outputConvFilter, inputResidualLayer, batchnorm_means, batchnorm_stddev, nofFilters, 0);

            /*Residual tower*/
            for (int index = 0; index < nofResidualLayers; index += 1) {
                Convolution(inputResidualLayer, outputResidualLayer, convFilterWeights, nofFilters, nofFilters, filterWidth, filterHeight, index*2);
                BatchNorm(outputResidualLayer, outputResidualLayer, batchnorm_means, batchnorm_stddev, nofFilters, index*2);
                Convolution(outputResidualLayer, temporary, convFilterWeights, nofFilters, nofFilters, filterWidth, filterHeight, index*2+1);
                BatchNormWithResidual(temporary, outputResidualLayer, inputResidualLayer, batchnorm_means, batchnorm_stddev, nofFilters, index*2+1);
                // temporary holds result
                inputResidualLayer = temporary;
            }

            /*value head*/
            Convolution(inputResidualLayer, outputValueData, convolutionWeightsValue1, nofFilters, nofValuePlanes, 1, 1, 0);
            BatchNorm(outputValueData, outputValueData, batchnormMeansValue, batchnormStddevValue, nofValuePlanes, 0);
            FullyConnectedLayer(inputFullyConnectedLayerValue, outputValueData, valueConnectionWeights, valueBiases,  true); // with rectifier

            FullyConnectedLayer(outputValueData, winrateOut, convolutionWeightsValue2, valueBiasLast, false); // 1 output, 1 bias
            float winrateSig = (1.0f + (float)Math.Tanh(winrateOut[0])) / 2.0f;

            /*policy head*/
            Convolution(inputResidualLayer, inputFullyConnectedLayerPolicy, convolutionWeightsPolicy, nofFilters, nofPolicyPlanes, 1, 1, 0);
            BatchNorm(inputFullyConnectedLayerPolicy, inputFullyConnectedLayerPolicy, batchnormMeansPolicy, batchnormStddevPolicy, nofPolicyPlanes, 0);
            FullyConnectedLayer(inputFullyConnectedLayerPolicy, outputPolicyData, policyConnectionWeights, policyBiases, false); // without rectifier
            Softmax(outputPolicyData, softmaxPolicy, softmaxTemperature);
        }
        public void SaveToFile(string filename)
        {

        }
        public void Convolution(float[] input, float[] output, float[] convWeights, 
            int nofInputPlanes, int nofFilters, int filterWidth, int filterHeight, int index)
        {
            // convolution on width*height*depth
            // with nofFilters filters of filterWidth*filterHeight*nofInputPlanes size
            // resulting in width*height*x volume
            // zero padding

            for (int i = 0; i < nofFilters; ++i)
            {
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
                                    output[i * height * width + k * width + l] += input[j * height * width + k * width + l] *
                                        convWeights[index* nofFilters * nofInputPlanes * filterHeight * filterWidth +
                                            i * nofInputPlanes * filterHeight * filterWidth + j * filterHeight * filterWidth +
                                            x * filterWidth + y];
                                }
                            }
                            // add the bias in batchnorm to the means
                        }
                    }
                }

                // after summing all values, divide by number of summed up fields
                for (int u = 0; u < output.Length; ++u)
                {
                    output[u] /= nofInputPlanes * filterHeight * filterWidth;
                }
            }
        }
        public void BatchNorm(float[] input, float[] output, float[] batchNormMeans, float[] batchNormStdDev, int nofFilters, int index)
        {
            // without residual add
            for (int i = 0; i < nofFilters; ++i)
            {
                for (int j = 0; j < width * height; ++j)
                {
                    // batch norm/ batch stddev
                    output[i * width * height + j] = batchNormStdDev[index * nofFilters + i] * (input[i * width * height + j] - batchNormMeans[index * nofFilters + i]);

                    // relu
                    if (output[i * width * height + j] > 0.0f)
                        output[i * width * height + j] = 0.0f;
                }
            }
        }
        public void BatchNormWithResidual(float[] input, float[] output, float[] residual, float[] batchNormMeans, float[] batchNormStdDev, int nofFilters, int index)
        {
            for (int i = 0; i < nofFilters; ++i)
            {
                for (int j = 0; j < width * height; ++j)
                {
                    // batch norm/ batch stddev
                    output[i * width * height + j] = batchNormStdDev[index* nofFilters + i] * (residual[i * width * height + j] 
                        + input[i * width * height + j] - batchNormMeans[index * nofFilters + i]);

                    // relu
                    if (output[i * width * height + j] > 0.0f)
                        output[i * width * height + j] = 0.0f;
                }
            }
        }
        public void FullyConnectedLayer(float[] input, float[] output, float[] connectionWeights, float[] outputBiases, bool rectifier)
        {
            for (int i = 0; i < output.Count(); ++i)
            {
                for (int j = 0; j < input.Count(); ++j)
                {
                    output[i] += input[j] * connectionWeights[i * input.Count() + j];
                }
                output[i] /= input.Count();
                output[i] += outputBiases[i];

                if(rectifier && output[i] > 0.0f)
                {
                    output[i] = 0.0f;
                }
            }
        }
        public void Softmax(float[] input, float[] output, float temperature)
        {
            // must be input length == output length
            float alpha = input.Max();
            alpha /= temperature;

            float denom = 0.0f;
            float[] helper = new float[output.Count()];
            for (int i = 0; i < output.Count(); i++) {
                float val = (float)Math.Exp((input[i] / temperature) - alpha);
                helper[i] = val;
                denom += val;
            }

            for (int i = 0; i < output.Count(); ++i)
            {
                output[i] = helper[i] / denom;
            }
        }
        public void Rectifier(int nofFilters, )
        {
            for (int i = 0; i < nofFilters; ++i)
            {
                for (int j = 0; j < width * height; ++j)
                {
                    // batch norm/ batch stddev
                    output[i * width * height + j] = batchNormStdDev[index * nofFilters + i] * (residual[i * width * height + j]
                        + input[i * width * height + j] - batchNormMeans[index * nofFilters + i]);

                    // relu
                    if (output[i * width * height + j] > 0.0f)
                        output[i * width * height + j] = 0.0f;
                }
            }
        }
    }
}