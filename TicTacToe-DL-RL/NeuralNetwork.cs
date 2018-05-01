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
        float[] firstConvFilterWeights = new float[nofFilters* nofPlanes * filterHeight * filterWidth]; // weights

        // for residual tower
        float[] inputResidualLayer = new float[nofFilters * height * width]; // input to residual layer
        float[] outputResidualLayer = new float[nofFilters * height * width];
        float[] temporary = new float[nofFilters * height * width];
        float[] convFilterWeights = new float[(nofConvLayers-1) * nofFilters * nofFilters * filterHeight * filterWidth]; // weights

        // for policy layer
        float[] convolutionWeightsPolicy = new float[nofPolicyPlanes* nofFilters]; // weights
        float[] convolutionBiasesPolicy = new float[nofPolicyPlanes* nofFilters]; // weights
        float[] batchnormMeansPolicy = new float[nofPolicyPlanes]; // weights
        float[] batchnormStddevPolicy = new float[nofPolicyPlanes]; // weights
        float[] policyConnectionWeights = new float[width * height * nofPolicyPlanes * nofOutputPolicies]; // weights
        float[] policyBiases = new float[nofOutputPolicies]; // weights
        float[] inputFullyConnectedLayerPolicy = new float[nofPolicyPlanes * width * height];
        float[] outputPolicyData = new float[nofOutputPolicies];

        // for value layer
        float[] convolutionWeightsValue1 = new float[nofFilters* nofValuePlanes]; // 1x1 filters, 32 of them for 64 input planes // weights
        float[] convolutionWeightsValue2 = new float[128]; // weights
        float[] batchnormMeansValue = new float[nofValuePlanes]; // weights
        float[] batchnormStddevValue = new float[nofValuePlanes]; // weights
        float[] valueConnectionWeights = new float[width * height * nofValuePlanes * 128]; // weights
        float[] valueBiases = new float[128]; // weights
        float[] valueBiasLast = new float[1]; // weights
        float[] inputFullyConnectedLayerValue = new float[nofValuePlanes * width * height];
        float[] outputValueData = new float[nofValuePlanes * width * height];
        float[] temporaryValueData = new float[128];

        // for all
        float[] convBiases = new float[nofConvLayers * nofFilters]; // weights
        float[] batchnorm_means = new float[nofConvLayers * nofFilters];
        float[] batchnorm_stddev = new float[nofConvLayers * nofFilters]; // weights

        // output of NN
        float[] softmaxPolicy = new float[nofOutputPolicies];
        float[] winrateOut = new float[1];

        public NeuralNetwork()
        {
            SaveWeightsToFile("weights.txt");
            ReadWeightsFromFile("weights.txt");
            //ReadWeightsFromFile("weights.txt");
            //for (int i = 0; i < convBiases.Count(); ++i) {
            //    batchnorm_means[i] -= convBiases[i];
            //    convBiases[i] = 0.0f;
            //}
        }
        public Tuple<float[], float> Predict(Position pos)
        {
            // returns array of move evals and V
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
                input[9 + i] = gameBoard[i];
            }
            return ForwardPassCPU(input);
        }
        public Tuple<float[], float> ForwardPassCPU(float[] input)
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
                float[] swap = inputResidualLayer;
                inputResidualLayer = outputResidualLayer;
                outputResidualLayer = swap;
            }

            /*value head*/
            Convolution(inputResidualLayer, outputValueData, convolutionWeightsValue1, nofFilters, nofValuePlanes, 1, 1, 0);
            BatchNorm(outputValueData, outputValueData, batchnormMeansValue, batchnormStddevValue, nofValuePlanes, 0);
            FullyConnectedLayer(outputValueData, temporaryValueData, valueConnectionWeights, valueBiases,  true); // with rectifier
            Rectifier(temporaryValueData);
            FullyConnectedLayer(temporaryValueData, winrateOut, convolutionWeightsValue2, valueBiasLast, false); // 1 output, 1 bias
            float winrateSig = (1.0f + (float)Math.Tanh(winrateOut[0])) / 2.0f;

            /*policy head*/
            Convolution(inputResidualLayer, inputFullyConnectedLayerPolicy, convolutionWeightsPolicy, nofFilters, nofPolicyPlanes, 1, 1, 0);
            BatchNorm(inputFullyConnectedLayerPolicy, inputFullyConnectedLayerPolicy, batchnormMeansPolicy, batchnormStddevPolicy, nofPolicyPlanes, 0);
            FullyConnectedLayer(inputFullyConnectedLayerPolicy, outputPolicyData, policyConnectionWeights, policyBiases, false); // without rectifier
            Softmax(outputPolicyData, softmaxPolicy, softmaxTemperature);

            return Tuple.Create(softmaxPolicy, winrateSig);

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
        public void Rectifier(float[] data)
        {
            for (int i = 0; i < data.Count(); ++i)
            { 
                // relu
                if (data[i] > 0.0f)
                    data[i] = 0.0f;
            }
        }
        public void ReadWeightsFromFile(string name)
        {
            String text;
            String[] tokens;

            StreamReader streamReader = new StreamReader(name);

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                firstConvFilterWeights[i] = float.Parse(tokens[i]);
            }


            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                convFilterWeights[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                convBiases[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                batchnorm_means[i] = float.Parse(tokens[i]);
            }


            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                batchnorm_stddev[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                convolutionWeightsPolicy[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                convolutionBiasesPolicy[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                batchnormMeansPolicy[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                batchnormStddevPolicy[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                policyConnectionWeights[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                policyBiases[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                convolutionWeightsValue1[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                convolutionWeightsValue2[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                batchnormMeansValue[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                batchnormStddevValue[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                valueConnectionWeights[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                valueBiases[i] = float.Parse(tokens[i]);
            }

            text = streamReader.ReadLine();
            tokens = text.Split(' ');
            for (int i = 0; i < tokens.Count(); ++i)
            {
                valueBiasLast[i] = float.Parse(tokens[i]);
            }
        }
        public void SaveWeightsToFile(string name)
        {
            StreamWriter fileWriter = new StreamWriter(name);

            for (int i = 0; i < firstConvFilterWeights.Count() - 1; ++i)
            {
                fileWriter.Write(firstConvFilterWeights[i] + " ");
            }
            fileWriter.Write(firstConvFilterWeights[firstConvFilterWeights.Count()-1] + "\n");

            for (int i = 0; i < convFilterWeights.Count()-1; ++i)
            {
                fileWriter.Write(convFilterWeights[i] + " ");
            }
            fileWriter.Write(convFilterWeights[convFilterWeights.Count() - 1] + "\n");

            for (int i = 0; i < convBiases.Count()-1; ++i)
            {
                fileWriter.Write(convBiases[i] + " ");
            }
            fileWriter.Write(convBiases[convBiases.Count() - 1] + "\n");

            for (int i = 0; i < batchnorm_means.Count()-1; ++i)
            {
                fileWriter.Write(batchnorm_means[i] + " ");
            }
            fileWriter.Write(batchnorm_means[batchnorm_means.Count() - 1] + "\n");

            //w = 1.0f / std::sqrt(w + epsilon) on read of bn stddivs
            for (int i = 0; i < batchnorm_stddev.Count()-1; ++i)
            {
                fileWriter.Write(batchnorm_stddev[i] + " ");
            }
            fileWriter.Write(batchnorm_stddev[batchnorm_stddev.Count() - 1] + "\n");

            // policy head
            for (int i = 0; i < convolutionWeightsPolicy.Count()-1; ++i)
            {
                fileWriter.Write(convolutionWeightsPolicy[i] + " ");
            }
            fileWriter.Write(convolutionWeightsPolicy[convolutionWeightsPolicy.Count() - 1] + "\n");

            for (int i = 0; i < convolutionBiasesPolicy.Count()-1; ++i)
            {
                fileWriter.Write(convolutionBiasesPolicy[i] + " ");
            }
            fileWriter.Write(convolutionBiasesPolicy[convolutionBiasesPolicy.Count() - 1] + "\n");

            for (int i = 0; i < batchnormMeansPolicy.Count()-1; ++i)
            {
                fileWriter.Write(batchnormMeansPolicy[i] + " ");
            }
            fileWriter.Write(batchnormMeansPolicy[batchnormMeansPolicy.Count() - 1] + "\n");

            for (int i = 0; i < batchnormStddevPolicy.Count()-1; ++i)
            {
                fileWriter.Write(batchnormStddevPolicy[i] + " ");
            }
            fileWriter.Write(batchnormStddevPolicy[batchnormStddevPolicy.Count() - 1] + "\n");

            for (int i = 0; i < policyConnectionWeights.Count()-1; ++i)
            {
                fileWriter.Write(policyConnectionWeights[i] + " ");
            }
            fileWriter.Write(policyConnectionWeights[policyConnectionWeights.Count() - 1] + "\n");

            for (int i = 0; i < policyBiases.Count()-1; ++i)
            {
                fileWriter.Write(policyBiases[i] + " ");
            }
            fileWriter.Write(policyBiases[policyBiases.Count() - 1] + "\n");

            // for value
            for (int i = 0; i < convolutionWeightsValue1.Count()-1; ++i)
            {
                fileWriter.Write(convolutionWeightsValue1[i] + " ");
            }
            fileWriter.Write(convolutionWeightsValue1[convolutionWeightsValue1.Count() - 1] + "\n");

            for (int i = 0; i < convolutionWeightsValue2.Count()-1; ++i)
            {
                fileWriter.Write(convolutionWeightsValue2[i] + " ");
            }
            fileWriter.Write(convolutionWeightsValue2[convolutionWeightsValue2.Count() - 1] + "\n");

            for (int i = 0; i < batchnormMeansValue.Count()-1; ++i)
            {
                fileWriter.Write(batchnormMeansValue[i] + " ");
            }
            fileWriter.Write(batchnormMeansValue[batchnormMeansValue.Count() - 1] + "\n");

            for (int i = 0; i < batchnormStddevValue.Count()-1; ++i)
            {
                fileWriter.Write(batchnormStddevValue[i] + " ");
            }
            fileWriter.Write(batchnormStddevValue[batchnormStddevValue.Count() - 1] + "\n");

            for (int i = 0; i < valueConnectionWeights.Count()-1; ++i)
            {
                fileWriter.Write(valueConnectionWeights[i] + " ");
            }
            fileWriter.Write(valueConnectionWeights[valueConnectionWeights.Count() - 1] + "\n");

            for (int i = 0; i < valueBiases.Count()-1; ++i)
            {
                fileWriter.Write(valueBiases[i] + " ");
            }
            fileWriter.Write(valueBiases[valueBiases.Count() - 1] + "\n");

            for (int i = 0; i < valueBiasLast.Count()-1; ++i)
            {
                fileWriter.Write(valueBiasLast[i] + " ");
            }
            fileWriter.Write(valueBiasLast[valueBiasLast.Count() - 1] + "\n");
            fileWriter.Close();
        }
    }
}