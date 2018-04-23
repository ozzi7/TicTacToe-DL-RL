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
        const int nofFilters = 64;
        const int nofConvLayers = 13;
        const int nofResidualLayers = 6; // half of (conv-1), 1 conv layer is for input (heads are seperate)

        float[] inputFirstConvFilter = new float[nofPlanes * height * width];
        float[] outputConvFilter = new float[nofFilters * height * width];
        // for input
        float[] firstConvFilterWeights = new float[nofFilters* nofPlanes * filterHeight * filterWidth];
        // for res tower
        float[] convFilterWeights = new float[(nofConvLayers-1) * filterHeight * filterWidth];
        float[] convBiases = new float[nofConvLayers * nofFilters];
        float[] batchnorm_means = new float[nofConvLayers * nofFilters];
        float[] batchnorm_stddev = new float[nofConvLayers * nofFilters];

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
            List<float> outputValueData = new List<float>(nofPlanes* width * height);
            List<float> outputPolicyData = new List<float>(nofOutputPolicies);
            List<float> softmaxData = new List<float>(nofOutputPolicies);

            // set nn input
            for (int i = 0; i < 9; ++i)
            {   // whose turn it is
                inpuData.Add(pos.sideToMove);
            }
            for (int i = 0; i < 9; ++i)
            {   // the board itself
                inpuData.Add(gameBoard[i]);
            }
            ForwardPassCPU(inpuData, outputPolicyData, outputValueData);
            return null;
        }
        public void ForwardPassCPU(List<float> inputData, List<float> outputPolicyData, List<float> outputValueData)
        {
            float[] input = inputData.ToArray();
            Convolution(input, outputConvFilter, nofPlanes);
            BatchNorm(outputConvFilter);
        }
        public void SaveToFile(string filename)
        {

        }
        public void Convolution(float[] input, float[] output, int nofInputPlanes)
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
                                    outputConvFilter[i * height * width + k * width + l] +=
                                        input[j * height * width + k * width + l] * 
                                        firstConvFilterWeights[i* nofInputPlanes * filterHeight*filterWidth + j*filterHeight*filterWidth +
                                        x * filterWidth + y];
                                }
                            }
                            // add the bias in batchnorm to the means
                        }
                    }
                }

                // after summing all values, divide by number of summed up fields
                for (int u = 0; u < outputConvFilter.Length; ++u)
                {
                    outputConvFilter[u] /= nofInputPlanes * filterHeight * filterWidth;
                }
            }
        }
        public void BatchNorm(float[] data)
        {
            // for first input
            for (int i = 0; i < nofFilters; ++i)
            {
                for (int j = 0; j < width * height; ++j)
                {
                    // batch norm/ batch stddev
                    data[i * width * height + j] = batchnorm_stddev[i] * (data[i * width * height + j] - batchnorm_means[i]);

                    // relu
                    if (data[i * width * height + j] > 0.0f)
                        data[i * width * height + j] = 0.0f;
                }
            }
        }
        public void BatchNormWithResidual(float[] data, float[] res)
        {
            // for first input
            for (int i = 0; i < nofFilters; ++i)
            {
                for (int j = 0; j < width * height; ++j)
                {
                    // batch norm/ batch stddev
                    data[i * width * height + j] = batchnorm_stddev[i] * (res[i * width * height + j] 
                        + data[i * width * height + j] - batchnorm_means[i]);

                    // relu
                    if (data[i * width * height + j] > 0.0f)
                        data[i * width * height + j] = 0.0f;
                }
            }
        }
    }
}