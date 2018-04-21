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
        const int nofPolicyInputPlanes = 32; // for some reason we only want 32 planes in policy/value heads (the input to is 64 and
        const int nofValueInputPlanes = 32; // conv makes it 32)

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
        float[] convolutionWeightsPolicy = new float[nofPolicyInputPlanes];
        float[] convolutionBiasesPolicy = new float[nofPolicyInputPlanes];

        // for value layer
        float[] convolutionWeightsValue = new float[nofValueInputPlanes];
        float[] convolutionBiasesValue = new float[nofValueInputPlanes];

        // for all
        float[] convBiases = new float[nofConvLayers * nofFilters];
        float[] batchnorm_means = new float[nofConvLayers * nofFilters];
        float[] batchnorm_stddev = new float[nofConvLayers * nofFilters];

        // output of NN
        float[] outputValueData = new float[nofPlanes * width * height];
        float[] outputPolicyData = new float[nofOutputPolicies];

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
            ForwardPassCPU(input, outputPolicyData, outputValueData);
            return null;
        }
        public void ForwardPassCPU(float[] input, float[] outputPolicyData, float[] outputValueData)
        {
            /*Conv layer */
            Convolution(input, outputConvFilter, firstConvFilterWeights, nofPlanes, nofFilters, filterWidth, filterHeight);
            BatchNorm(outputConvFilter, inputResidualLayer, batchnorm_means, batchnorm_stddev, nofFilters);

            /*Residual tower*/
            for (int i = 0; i < nofResidualLayers; i += 1) {
                ResLayerConvolution(inputResidualLayer, outputResidualLayer, nofFilters);
                BatchNorm(outputResidualLayer, outputResidualLayer, batchnorm_means, batchnorm_stddev, nofFilters);
                ResLayerConvolution(outputResidualLayer, temporary, nofFilters);
                BatchNormWithResidual(temporary, inputResidualLayer);
                // temporary holds result
                inputResidualLayer = temporary;
            }

            /*value head*/
            ConvolutionPolicy(inputResidualLayer, outputPolicyData);
            BatchNorm(outputResidualLayer, outputResidualLayer);
            //convolve < 1 > (NUM_VALUE_INPUT_PLANES, conv_out, conv_val_w, conv_val_b, value_data);
            batchnorm < width * height > (NUM_VALUE_INPUT_PLANES, value_data, bn_val_w1.data(), bn_val_w2.data());
            /*policy head*/
            convolve < 1 > (NUM_POLICY_INPUT_PLANES, conv_out, conv_pol_w, conv_pol_b, policy_data);
            batchnorm < width * height > (NUM_POLICY_INPUT_PLANES, policy_data, bn_pol_w1.data(), bn_pol_w2.data());


            innerproduct < NUM_POLICY_INPUT_PLANES * width * height, NUM_OUTPUT_POLICY > (policy_data, ip_pol_w, ip_pol_b, output_pol);
            innerproduct < NUM_VALUE_INPUT_PLANES * width * height, NUM_VALUE_CHANNELS > (value_data, ip1_val_w, ip1_val_b, output_val);
        }
        public void SaveToFile(string filename)
        {

        }
        public void Convolution2(float[] input, float[] output)
        {
            // convolution on width*height*depth
            // with nofFilters filters of filterWidth*filterHeight*nofInputPlanes size
            // resulting in width*height*x volume
            // zero padding

            for (int i = 0; i < nofFilters; ++i)
            {
                // apply each of the filters to the complete input..
                for (int j = 0; j < nofPlanes; ++j)
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
                                        firstConvFilterWeights[i* nofPlanes * filterHeight*filterWidth + j*filterHeight*filterWidth +
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
                    output[u] /= nofPlanes * filterHeight * filterWidth;
                }
            }
        }
        public void Convolution(float[] input, float[] output, float[] convWeights, 
            int nofInputPlanes, int nofFilters, int filterWidth, int filterHeight)
        {
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
                                    output[i * height * width + k * width + l] +=
                                        input[j * height * width + k * width + l] *
                                        convWeights[i * nofInputPlanes * filterHeight * filterWidth + j * filterHeight * filterWidth +
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
        public void BatchNorm(float[] input, float[] output, float[] batchNormMeans, float[] batchNormStdDev, int nofFilters)
        {
            // without residual add
            for (int i = 0; i < nofFilters; ++i)
            {
                for (int j = 0; j < width * height; ++j)
                {
                    // batch norm/ batch stddev
                    output[i * width * height + j] = batchNormStdDev[i] * (input[i * width * height + j] - batchNormMeans[i]);

                    // relu
                    if (output[i * width * height + j] > 0.0f)
                        output[i * width * height + j] = 0.0f;
                }
            }
        }
        public void BatchNormWithResidual(float[] input, float[] output, float[] residual, float[] batchNormMeans, float[] batchNormStdDev, int nofFilters)
        {
            for (int i = 0; i < nofFilters; ++i)
            {
                for (int j = 0; j < width * height; ++j)
                {
                    // batch norm/ batch stddev
                    output[i * width * height + j] = batchNormStdDev[i] * (residual[i * width * height + j] 
                        + input[i * width * height + j] - batchNormMeans[i]);

                    // relu
                    if (output[i * width * height + j] > 0.0f)
                        output[i * width * height + j] = 0.0f;
                }
            }
        }
    }
}