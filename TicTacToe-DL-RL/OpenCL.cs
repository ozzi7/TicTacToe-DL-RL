/*
 * Idea: Create buffers for the network weights and keep X different networks in GPU memory
 * For each network we also add Y different inputs and so we run X*Y forward passes in one call
 * The inputs are changed on each call but weights only change after evolution
 */
 using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Cloo;
using System.IO;

namespace TicTacToe_DL_RL
{
    static class OpenCL
    {
        static int NOF_PARALLEL_EXECUTIONS = 10;

        // for input layer
        public static List<float> input = new List<float>();
        public static List<float> outputConvFilter = new List<float>();
        public static List<float> firstConvFilterWeights = new List<float>();

        // for residual tower
        public static List<float> inputResidualLayer = new List<float>();
        public static List<float> outputResidualLayer = new List<float>();
        public static List<float> temporary = new List<float>();
        public static List<float> convFilterWeights = new List<float>();

        // for policy layer
        public static List<float> convWeightsPolicy = new List<float>();
        public static List<float> convBiasesPolicy = new List<float>();
        public static List<float> BNMeansPolicy = new List<float>();
        public static List<float> BNStddevPolicy = new List<float>();
        public static List<float> BNBetaPolicy = new List<float>();
        public static List<float> BNGammaPolicy = new List<float>();
        public static List<float> policyConnectionWeights = new List<float>();
        public static List<float> policyBiases = new List<float>();
        public static List<float> inputFCLayerPolicy = new List<float>();
        public static List<float> outputPolicyData = new List<float>();

        // for value layer
        public static List<float> convWeightsValue1 = new List<float>();
        public static List<float> convWeightsValue2 = new List<float>();
        public static List<float> BNMeansValue = new List<float>();
        public static List<float> BNStddevValue = new List<float>();
        public static List<float> BNBetaValue = new List<float>();
        public static List<float> BNGammaValue = new List<float>();
        public static List<float> valueConnectionWeights = new List<float>();
        public static List<float> valueBiases = new List<float>();
        public static List<float> valueBiasLast = new List<float>();
        public static List<float> inputFCLayerValue = new List<float>();
        public static List<float> outputValueData = new List<float>();
        public static List<float> temporaryValueData = new List<float>();

        // for all layers
        public static List<float> convBiases = new List<float>();
        public static List<float> BNMeans = new List<float>();
        public static List<float> BNStddev = new List<float>();

        public static List<float> BNBetas = new List<float>();
        public static List<float> BNGammas = new List<float>();

        // output of NN
        public static List<float> softmaxPolicy = new List<float>();
        public static List<float> winrateOut = new List<float>();

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

        // opencl stuff
        static private ComputeProgram program;
        static private ComputePlatform platform;
        static private ComputeContext context;
        static private ComputeContextPropertyList properties;
        static private ComputeKernel kernel;

        public static void CompileKernel()
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

            // Create the kernel function and set its arguments.
            kernel = program.CreateKernel("NN");

            // The output buffer doesn't need any data from the host. Only its size is specified res.length.
            CB_results = new ComputeBuffer<float>(context, ComputeMemoryFlags.WriteOnly | ComputeMemoryFlags.CopyHostPointer, 26* NOF_PARALLEL_EXECUTIONS);

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

        }
        /* resend all weights, later we will re-use the weights (for example keep 10 different neural network weights in GPU and re-use them)*/
        public static Tuple<float[], float> EnqueueWork(NeuralNetwork nn)
        {
            input.AddRange(nn.input);
        

        }
        public static void RunKernels()
        {
            CB_input = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, input.ToArray());
            float[] output = new float[26*10];

            try
            {
                // Create the event wait list. An event list is not really needed for this example but it is important to see how it works.
                // Note that events (like everything else) consume OpenCL resources and creating a lot of them may slow down execution.
                // For this reason their use should be avoided if possible.
                ComputeEventList eventList = new ComputeEventList();

                // Create the command queue. This is used to control kernel execution and manage read/write/copy operations.
                ComputeCommandQueue commands = new ComputeCommandQueue(context, context.Devices[0], ComputeCommandQueueFlags.None);

                // Execute the kernel "count" times. After this call returns, "eventList" will contain an event associated with this command.
                // If eventList == null or typeof(eventList) == ReadOnlyCollection<ComputeEventBase>, a new event will not be created.

                commands.Execute(kernel, null, new long[] { 1 }, null, eventList);

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

            return Tuple.Create(policy, output[25]);
        }
        public static void CreateNetworkWeightBuffers()
        {
            // Create the input buffers and fill them with data from the arrays.
            // Access modifiers should match those in a kernel.
            // CopyHostPointer means the buffer should be filled with the data provided in the last argument.
            // opencl buffers
            CB_firstConvFilterWeights = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, firstConvFilterWeights.ToArray());

            CB_convBiases = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, convBiases.ToArray());
            CB_BNMeans = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNMeans.ToArray());
            CB_BNStddev = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNStddev.ToArray());
            CB_BNBetas = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNBetas.ToArray());
            CB_BNGammas = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNGammas.ToArray());

            CB_convWeightsValue1 = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, convWeightsValue1.ToArray());
            CB_convWeightsValue2 = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, convWeightsValue2.ToArray());
            CB_BNMeansValue = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNMeansValue.ToArray());
            CB_BNStddevValue = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNStddevValue.ToArray());
            CB_BNBetaValue = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNBetaValue.ToArray());
            CB_BNGammaValue = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNGammaValue.ToArray());
            CB_valueConnectionWeights = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, valueConnectionWeights.ToArray());
            CB_valueBiases = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, valueBiases.ToArray());
            CB_valueBiasLast = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, valueBiasLast.ToArray());
            CB_inputFCLayerValue = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, inputFCLayerValue.ToArray());

            CB_convWeightsPolicy = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, convWeightsPolicy.ToArray());
            CB_convBiasesPolicy = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, convBiasesPolicy.ToArray());
            CB_BNMeansPolicy = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNMeansPolicy.ToArray());
            CB_BNStddevPolicy = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNStddevPolicy.ToArray());
            CB_BNBetaPolicy = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNBetaPolicy.ToArray());
            CB_BNGammaPolicy = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, BNGammaPolicy.ToArray());
            CB_policyConnectionWeights = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, policyConnectionWeights.ToArray());
            CB_policyBiases = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, policyBiases.ToArray());
            CB_convFilterWeights = new ComputeBuffer<float>(context, ComputeMemoryFlags.ReadOnly | ComputeMemoryFlags.CopyHostPointer, convFilterWeights.ToArray());

            try
            {
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
            catch (Exception e)
            {
                Console.WriteLine(e.ToString());
            }
        }
    }
}
