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
        const int nofPlanes = 2; // input channels, board 9x9 + color 9x9
        const int nofOutputPolicies = 9; // policy net has 9 outputs (1 per potential move)
        const int nofOutputValues = 1; // value head has 1 output
        const int nofFilters = 64;
        const int nofConvLayers = 26;

        float[] inputFirstConvFilter = new float[nofPlanes * height * width];
        float[] outputConvFilter = new float[nofFilters * height * width];
        float[] firstConvFilterWeights = new float[nofPlanes * filterHeight * filterWidth];
        float[] convBiases = new float[nofConvLayers * nofFilters];

        public NeuralNetwork()
        {
            /*init some arrays with random values */
        }

        public Tuple<List<float>, float> Predict(Position pos)
        {            
            /*Not using game history, not using caching*/
            int[] tmp = new int[pos.gameBoard.GetLength(0) * pos.gameBoard.GetLength(1)];
            Buffer.BlockCopy(pos.gameBoard, 0, tmp, 0, tmp.Length * sizeof(int));
            List<int> gameBoard = new List<int>(tmp);

            List<float> inpuData = new List<float>(width * height * nofPlanes);
            List<float> outputValueData = new List<float>(width * height * nofPlanes);
            List<float> outputPolicyData = new List<float>(nofOutputPolicies);
            List<float> softmaxData = new List<float>(nofOutputPolicies);

            // set nn input
            for(int i = 0; i < 9; ++i)
            {   // whose turn it is
                inpuData.Add(pos.sideToMove);
            }
            for(int i = 0; i < 9; ++i)
            {   // the board itself
                inpuData.Add(gameBoard[i]);
            }
            ForwardPassCPU(inpuData, outputPolicyData, outputValueData);
            return null;
        }
        public void ForwardPassCPU(List<float> inputData, List<float> outputPolicyData, List<float> outputValueData)
        {
            float[] input = inputData.ToArray();
            Convolution(input, outputConvFilter);
        }
        public void SaveToFile(string filename)
        {

        }
        public void Convolution(float[] input, float[] output)
        {
            // convolution on width*height*depth
            // with nofFilters filters of filterWidth*filterHeight*nofPlanes size
            // resulting in width*height*x volume
            // zero padding

            for(int i = 0; i < nofFilters; ++i)
            {
                // apply each of the filters..
                for (int j = 0; j < nofPlanes; ++j)
                {
                    for (int k = 0; k < height; ++k)
                    {
                        for (int l = 0; l < width; ++l)
                        {
                            // looking at a 1x1x1 of the input here
                            for (int x = 0; x < filterHeight; ++x)
                            {
                                for (int y = 0; y < filterWidth; ++y)
                                {
                                    // going through the neighbors
                                    if(k - filterHeight / 2 + x < 0 || k - filterHeight / 2 + x >= height ||
                                        l - filterWidth / 2 + y < 0 || l - filterWidth / 2 + y >= width)
                                    {
                                        // the filter is out of bounds, set to 0 (0 padding)
                                        continue;
                                    }
                                    outputConvFilter[i * height * width+k*width+l] +=
                                        input[j * height * width + k * width + l] * firstConvFilterWeights[x * filterWidth + y];
                                }
                            }
                            // after summing all values, divide by number of summed up fields
                            outputConvFilter[i * height * width + k * width + l] /= nofPlanes * filterHeight * filterWidth;
                            // add bias per output plane/per filter
                            outputConvFilter[i * height * width + k * width + l] += convBiases[i];
                        } 
                    }
                }
            }
        }
    }
}