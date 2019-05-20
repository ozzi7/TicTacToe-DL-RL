using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cloo;

namespace TicTacToe_DL_RL
{
    class NeuralNetwork
    {
        // params
        const int gameboardWidth = 5;
        const int gameboardHeight = 5;
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
        float[] input = new float[nofPlanes * gameboardHeight * gameboardWidth]; // input to complete NN
        float[] outputConvFilter = new float[nofFilters * gameboardHeight * gameboardWidth];
        float[] firstConvFilterWeights = new float[nofFilters* nofPlanes * filterHeight * filterWidth]; // weights

        // for residual tower
        float[] inputResidualLayer = new float[nofFilters * gameboardHeight * gameboardWidth]; // input to residual layer
        float[] outputResidualLayer = new float[nofFilters * gameboardHeight * gameboardWidth];
        float[] temporary = new float[nofFilters * gameboardHeight * gameboardWidth];
        float[] convFilterWeights = new float[(nofConvLayers-1) * nofFilters * nofFilters * filterHeight * filterWidth]; // weights

        // for policy layer
        float[] convWeightsPolicy = new float[nofPolicyPlanes* nofFilters]; // weights 1x1xnofplanes filters
        float[] convBiasesPolicy = new float[nofPolicyPlanes* nofFilters]; // weights
        float[] BNMeansPolicy = new float[nofPolicyPlanes]; // weights
        float[] BNStddevPolicy = new float[nofPolicyPlanes]; // weights
        float[] BNBetaPolicy = new float[nofPolicyPlanes];
        float[] BNGammaPolicy = new float[nofPolicyPlanes];
        float[] policyConnectionWeights = new float[gameboardHeight * gameboardWidth* nofPolicyPlanes * nofOutputPolicies]; // weights
        float[] policyBiases = new float[nofOutputPolicies]; // weights
        float[] inputFCLayerPolicy = new float[nofPolicyPlanes * gameboardHeight* gameboardWidth];
        float[] outputPolicyData = new float[nofOutputPolicies];

        // for value layer
        float[] convWeightsValue1 = new float[nofFilters* nofValuePlanes]; // 1x1 filters, 32 of them for 64 input planes // weights
        float[] convWeightsValue2 = new float[128]; // weights
        float[] BNMeansValue = new float[nofValuePlanes]; // weights
        float[] BNStddevValue = new float[nofValuePlanes]; // weights
        float[] BNBetaValue = new float[nofValuePlanes];
        float[] BNGammaValue = new float[nofValuePlanes];
        float[] valueConnectionWeights = new float[gameboardHeight * gameboardWidth*nofValuePlanes * 128]; // weights
        float[] valueBiases = new float[128]; // weights
        float[] valueBiasLast = new float[1]; // weights
        float[] inputFCLayerValue = new float[nofValuePlanes *gameboardHeight*gameboardWidth];
        float[] outputValueData = new float[nofValuePlanes * gameboardHeight * gameboardWidth];
        float[] temporaryValueData = new float[128];

        // for all layers
        float[] convBiases = new float[nofConvLayers * nofFilters]; // weights
        float[] BNMeans = new float[nofConvLayers * nofFilters]; // UNTRAINABLE
        float[] BNStddev = new float[nofConvLayers * nofFilters]; // UNTRAINABLE

        float[] BNBetas = new float[nofConvLayers*nofFilters];
        float[] BNGammas = new float[nofConvLayers* nofFilters];

        // output of NN
        float[] softmaxPolicy = new float[nofOutputPolicies];
        float[] winrateOut = new float[1];

        // opencl buffers
        static private ComputeBuffer<float> CB_input;
        static private ComputeBuffer<float> CB_firstConvFilterWeights;
        static private ComputeBuffer<float> CB_convBiases;
        static private ComputeBuffer<float> CB_BNMeans;
        static private ComputeBuffer<float> CB_BNStddev;
        static private ComputeBuffer<float> CB_BNBetas;
        static private ComputeBuffer<float> CB_BNGammas;

        static private ComputeBuffer<float> CB_convWeightsValue1;
        static private ComputeBuffer<float> CB_convWeightsValue2;
        static private ComputeBuffer<float> CB_BNMeansValue;
        static private ComputeBuffer<float> CB_BNStddevValue;
        static private ComputeBuffer<float> CB_BNBetaValue;
        static private ComputeBuffer<float> CB_BNGammaValue;
        static private ComputeBuffer<float> CB_valueConnectionWeights;
        static private ComputeBuffer<float> CB_valueBiases;
        static private ComputeBuffer<float> CB_valueBiasLast;
        static private ComputeBuffer<float> CB_inputFCLayerValue;

        static private ComputeBuffer<float> CB_convWeightsPolicy;
        static private ComputeBuffer<float> CB_convBiasesPolicy;
        static private ComputeBuffer<float> CB_BNMeansPolicy;
        static private ComputeBuffer<float> CB_BNStddevPolicy;
        static private ComputeBuffer<float> CB_BNBetaPolicy;
        static private ComputeBuffer<float> CB_BNGammaPolicy;
        static private ComputeBuffer<float> CB_policyConnectionWeights;
        static private ComputeBuffer<float> CB_policyBiases;

        static private ComputeBuffer<float> CB_convFilterWeights;
        static private ComputeBuffer<float> CB_results;

        // complete weights
        public List<float> weights = new List<float>();
        public List<float> untrainable_weights = new List<float>();

        // opencl stuff
        static private ComputeProgram program;
        static private ComputePlatform platform;
        static private ComputeContext context;
        static private ComputeContextPropertyList properties;
        static private ComputeKernel kernel;

        public NeuralNetwork()
        {
            InitializeWeights();
            CalculateVirtualBNs();
            CompileKernel();
            //SaveWeightsToFile("weights_best.txt");
            //ReadWeightsFromFile("weights.txt");
            //ReadWeightsFromFile("weights.txt");
            //for (int i = 0; i < convBiases.Length; ++i) {
            //    BNMeans[i] -= convBiases[i];
            //    convBiases[i] = 0.0f;
            //}
        }
        private void CompileKernel()
        {
            platform = ComputePlatform.Platforms[0]; // todo find amd gpus..
            IList<ComputeDevice> devices;
            devices = new List<ComputeDevice>();

            object[] availableDevices = new object[platform.Devices.Count];
            for (int i = 0; i < availableDevices.Length; i++)
                availableDevices[i] = platform.Devices[i].Name;

            properties = new ComputeContextPropertyList(platform);
            devices.Add(platform.Devices[0]);
            context = new ComputeContext(devices, properties, null, IntPtr.Zero);

            // Create the input buffers and fill them with data from the arrays.
            // Access modifiers should match those in a kernel.
            // CopyHostPointer means the buffer should be filled with the data provided in the last argument.
            // opencl buffers
            CB_input = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, input);
            CB_firstConvFilterWeights = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, firstConvFilterWeights);

            CB_convBiases = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, convBiases);
            CB_BNMeans = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNMeans);
            CB_BNStddev = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNStddev);
            CB_BNBetas = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNBetas);
            CB_BNGammas = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNGammas);

            CB_convWeightsValue1 = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, convWeightsValue1);
            CB_convWeightsValue2 = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, convWeightsValue2);
            CB_BNMeansValue = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNMeansValue);
            CB_BNStddevValue = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNStddevValue);
            CB_BNBetaValue = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNBetaValue);
            CB_BNGammaValue = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNGammaValue);
            CB_valueConnectionWeights = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, valueConnectionWeights);
            CB_valueBiases = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, valueBiases);
            CB_valueBiasLast = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, valueBiasLast);
            CB_inputFCLayerValue = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, inputFCLayerValue);

            CB_convWeightsPolicy = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, convWeightsPolicy);
            CB_convBiasesPolicy = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, convBiasesPolicy);
            CB_BNMeansPolicy = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNMeansPolicy);
            CB_BNStddevPolicy = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNStddevPolicy);
            CB_BNBetaPolicy = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNBetaPolicy);
            CB_BNGammaPolicy = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNGammaPolicy);
            CB_policyConnectionWeights = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, policyConnectionWeights);
            CB_policyBiases = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, policyBiases);
            CB_convFilterWeights = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, convFilterWeights);

            // The output buffer doesn't need any data from the host. Only its size is specified res.length.
            float[] results = new float[26];

            CB_results = new ComputeBuffer<float>(context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.CopyHostPointer, results);

            // Create and build the opencl program.
            StreamReader streamReader = new StreamReader("../../NeuralNetwork.cl");
            string openClCode = streamReader.ReadToEnd();
            streamReader.Close();
            program = new ComputeProgram(context, openClCode);
            //string log = program.GetBuildLog(devices[0]);
            ComputeProgramBuildStatus error = program.GetBuildStatus(context.Devices[0]);
            try
            {
                program.Build(null, null, null, IntPtr.Zero);
            }
            catch
            {
                Console.WriteLine(program.GetBuildLog(devices[0]));
                throw;
            }

            // Create the kernel function and set its arguments.
            kernel = program.CreateKernel("NN");
            kernel.SetMemoryArgument(0, CB_input);
            kernel.SetMemoryArgument(1, CB_firstConvFilterWeights);

            kernel.SetMemoryArgument(2, CB_convBiases);
            kernel.SetMemoryArgument(3, CB_BNMeans);
            kernel.SetMemoryArgument(4, CB_BNStddev);
            kernel.SetMemoryArgument(5, CB_BNBetas);
            kernel.SetMemoryArgument(6, CB_BNGammas);

            kernel.SetMemoryArgument(7, CB_convWeightsValue1);
            kernel.SetMemoryArgument(8, CB_convWeightsValue2);
            kernel.SetMemoryArgument(9, CB_BNMeansValue);
            kernel.SetMemoryArgument(10, CB_BNStddevValue);
            kernel.SetMemoryArgument(11, CB_BNBetaValue);
            kernel.SetMemoryArgument(12, CB_BNGammaValue);
            kernel.SetMemoryArgument(13, CB_valueConnectionWeights);
            kernel.SetMemoryArgument(14, CB_valueBiases);
            kernel.SetMemoryArgument(15, CB_valueBiasLast);
            kernel.SetMemoryArgument(16, CB_inputFCLayerValue);

            kernel.SetMemoryArgument(17, CB_convWeightsPolicy);
            kernel.SetMemoryArgument(18, CB_convBiasesPolicy);
            kernel.SetMemoryArgument(19, CB_BNMeansPolicy);
            kernel.SetMemoryArgument(20, CB_BNStddevPolicy);
            kernel.SetMemoryArgument(21, CB_BNBetaPolicy);
            kernel.SetMemoryArgument(22, CB_BNGammaPolicy);
            kernel.SetMemoryArgument(23, CB_policyConnectionWeights);
            kernel.SetMemoryArgument(24, CB_policyBiases);
            kernel.SetMemoryArgument(25, CB_convFilterWeights);

            kernel.SetMemoryArgument(26, CB_results);
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
            //return ForwardPassGPU();
            return ForwardPassCPU(input);
        }
        private void CalculateVirtualBNs()
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
            float winrateSig = (1.0f + (float)Math.Tanh(winrateOut[0])) / 2.0f;

            /*policy head*/
            Convolution(inputResidualLayer, inputFCLayerPolicy, convWeightsPolicy, nofFilters, nofPolicyPlanes, 1, 1, 0);
            BN(inputFCLayerPolicy, inputFCLayerPolicy, BNMeansPolicy, BNStddevPolicy, nofPolicyPlanes, 0, BNGammaPolicy, BNBetaPolicy);
            FCLayer(inputFCLayerPolicy, outputPolicyData, policyConnectionWeights, policyBiases, false); // without rectifier
            Softmax(outputPolicyData, softmaxPolicy, softmaxTemperature);

            return Tuple.Create(softmaxPolicy, winrateSig);
        }
        public Tuple<float[], float> ForwardPassGPU()
        {
            float[] output = new float[26];
            try
            {
                CB_input = new ComputeBuffer<float>(context, ComputeMemoryFlags.CopyHostPointer, input);
                kernel.SetMemoryArgument(0, CB_input);

                // Create the event wait list. An event list is not really needed for this example but it is important to see how it works.
                // Note that events (like everything else) consume OpenCL resources and creating a lot of them may slow down execution.
                // For this reason their use should be avoided if possible.
                ComputeEventList eventList = new ComputeEventList();

                // Create the command queue. This is used to control kernel execution and manage read/write/copy operations.
                ComputeCommandQueue commands = new ComputeCommandQueue(context, context.Devices[0], ComputeCommandQueueFlags.None);

                // Execute the kernel "count" times. After this call returns, "eventList" will contain an event associated with this command.
                // If eventList == null or typeof(eventList) == ReadOnlyCollection<ComputeEventBase>, a new event will not be created.

                commands.Execute(kernel, null, new long[] { Params.MAX_GPU_WIDGETS }, null, eventList);

                // Read back the results. If the command-queue has out-of-order execution enabled (default is off), ReadFromBuffer 
                // will not execute until any previous events in eventList (in our case only eventList[0]) are marked as complete 
                // by OpenCL. By default the command-queue will execute the commands in the same order as they are issued from the host.
                // eventList will contain two events after this method returns.
                commands.ReadFromBuffer(CB_results, ref output, false, eventList);

                // A blocking "ReadFromBuffer" (if 3rd argument is true) will wait for itself and any previous commands
                // in the command queue or eventList to finish execution. Otherwise an explicit wait for all the opencl commands 
                // to finish has to be issued before "arrC" can be used. 
                // This explicit synchronization can be achieved in two ways:

                // 1) Wait for the events in the list to finish,
                //eventList.Wait();

                // 2) Or simply use
                commands.Finish();
                CB_input.Dispose();
                commands.Dispose();
            }
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
            }

            float[] policy = new float[25];
            Array.Copy(output, policy, 25);
            return Tuple.Create(policy, outputConvFilter[25]);
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
            for (int i = 0; i < convBiases.Length; ++i)
            {
                convBiases[i] = weights[count];
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
            for (int i = 0; i < convBiasesPolicy.Length; ++i)
            {
                convBiasesPolicy[i] = weights[count];
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

            for (int i = 0; i < convBiases.Length-1; ++i)
            {
                fileWriter.Write(convBiases[i] + " ");
            }
            fileWriter.Write(convBiases[convBiases.Length - 1] + "\n");

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

            for (int i = 0; i < convBiasesPolicy.Length-1; ++i)
            {
                fileWriter.Write(convBiasesPolicy[i] + " ");
            }
            fileWriter.Write(convBiasesPolicy[convBiasesPolicy.Length - 1] + "\n");

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
            for (int i = 0; i < convBiases.Length; ++i)
            {
                convBiases[i] = RandomNr.GetGaussianFloat();
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
            for (int i = 0; i < convBiasesPolicy.Length; ++i)
            {
                convBiasesPolicy[i] = RandomNr.GetGaussianFloat();
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
 
 