﻿static inline void Convolution(int sizeOfOutput, float* input, float* output, constant float* convWeights,
    int nofInputPlanes, int nofFilters, int filterWidth, int filterHeight, int index, int networkOffset)
{
    // convolution on gameboard_width*5*depth
    // with nofFilters filters of filterWidth*filterHeight*nofInputPlanes size
    // resulting in gameboard_width*5*x volume
    // zero padding

    for (int u = 0; u < sizeOfOutput; ++u)
    {
        output[u] = 0.0f;
    }
    for (int i = 0; i < nofFilters; ++i) { 
        // apply each of the filters to the complete input..
        for (int j = 0; j < nofInputPlanes; ++j)
        {
            for (int k = 0; k < 5; ++k)
            {
                for (int l = 0; l < 5; ++l)
                {
                    // looking at a 1x1x1 of the input here, we sum up the 3x3 neighbors (depending on filter size)
                    for (int x = 0; x < filterHeight; ++x)
                    {
                        for (int y = 0; y < filterWidth; ++y)
                        {
                            // going through the neighbors
                            if (k - filterHeight / 2 + x < 0 || k - filterHeight / 2 + x >= 5 ||
                                l - filterWidth / 2 + y < 0 || l - filterWidth / 2 + y >= 5)
                            {
                                // the filter is out of bounds, set to 0 (0 padding)
                                continue;
                            }
                            output[i * 5 * 5 + k * 5 + l] += 
                                input[j * 5 * 5 + k * 5 + l] *
                                convWeights[
									networkOffset +
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
    // after summing all values, divide by number of summed up fields
    for (int u = 0; u < sizeOfOutput; ++u)
    {
        output[u] /= nofInputPlanes * filterHeight * filterWidth;
    }
}
static inline void BN(int sizeofInput, float* input, float* output, constant float* BNMeans, constant float* BNStdDev, 
	int nofFilters, int index, constant float* BNGammas, constant float* BNBetas, int networkIndex)
{
    // without residual add
    for (int i = 0; i < nofFilters; ++i)
    {
        // go through each plane coming into BN and apply to each element the means and stddev..
        for (int j = 0; j < sizeofInput/nofFilters; ++j)
        {
            // we know the size of one plane by dividing input through number of planes (input.length/noffilters)
            // batch norm/ batch stddev
            /* see Alg 1: https://arxiv.org/pdf/1502.03167.pdf */
            float x_til = (float)((input[i * sizeofInput/ nofFilters + j] - BNMeans[networkIndex +  index * nofFilters + i])/
                (sqrt(BNStdDev[networkIndex +  index * nofFilters + i]+0.0001f)));
            output[i * sizeofInput / nofFilters + j] = BNGammas[networkIndex + index * nofFilters + i] *
					x_til+BNBetas[networkIndex + index * nofFilters + i];

            // relu
            if (output[i * sizeofInput / nofFilters + j] < 0.0f)
                output[i* sizeofInput / nofFilters + j] = 0.0f;
        }
    }
}
static inline void BNWithResidual(int sizeofInput,float* input, float* output, float* residual, constant float* BNMeans, constant float* BNStdDev, 
	int nofFilters, int index, constant float* BNGammas, constant float* BNBetas, int networkIndex)
{
    for (int i = 0; i < nofFilters; ++i)
    {
        for (int j = 0; j < 5 * 5; ++j)
        {
            // batch norm/ batch stddev
            float x_til = (float)((input[i * sizeofInput/ nofFilters + j] + 
                residual[i * sizeofInput / nofFilters + j] - BNMeans[networkIndex + index * nofFilters + i]) /
                (sqrt(BNStdDev[networkIndex +index * nofFilters + i] + 0.01f)));

            output[i * sizeofInput / nofFilters + j] = BNGammas[networkIndex + index * nofFilters + i] * x_til + 
																		BNBetas[networkIndex + index * nofFilters + i];

            // relu
            if (output[i * sizeofInput / nofFilters + j] < 0.0f)
                output[i * sizeofInput / nofFilters + j] = 0.0f;
        }
    }
}
static inline void FCLayer(int sizeofInput, int sizeofOutput, float* input, float* output, constant float* connectionWeights, constant float* outputBiases, bool rectifier, int connectionWeightIndex, int outputBiasesIndex)
{
    for (int u = 0; u < sizeofOutput; ++u)
    {
        output[u] = 0.0f;
    }
    for (int i = 0; i < sizeofOutput; ++i)
    {
        for (int j = 0; j < sizeofInput; ++j)
        {
            output[i] += input[j] * connectionWeights[connectionWeightIndex + i * sizeofInput + j];
        }
        output[i] /= sizeofInput;
        output[i] += outputBiases[outputBiasesIndex + i];

        if(rectifier && output[i] < 0.0f)
        {
            output[i] = 0.0f;
        }
    }
}
static inline void Softmax(float* input, float* output, float temperature)
{
    // must be input length == output length
	float alpha = -FLT_MAX;	
	for(int i = 0; i < 25; ++i) {
		if(input[i] >= alpha) {
			alpha = input[i];
		}
	}
    alpha /= temperature;

    float denom = 0.0f;
    float helper[25];
    for (int i = 0; i < 25; i++) {
        float val = (float)exp((input[i] / temperature) - alpha);
        helper[i] = val;
        denom += val;
    }

    for (int i = 0; i < 25; ++i)
    {
        output[i] = helper[i] / denom;
    }
}
static inline void Rectifier(int sizeofData, float* data)
{
    for (int i = 0; i < sizeofData; ++i)
    { 
        // relu
        if (data[i] < 0.0f)
            data[i] = 0.0f;
    }
}

/* entry point */
kernel void NN(
// for input layer
	constant float* input,
	constant float* firstConvFilterWeights,
// shared among all layers
	constant float* BNMeans,
	constant float* BNStddev,
	constant float* BNBetas,
	constant float* BNGammas,
// for value layer
	constant float* convWeightsValue1,
	constant float* convWeightsValue2,
	constant float* BNMeansValue,
	constant float* BNStddevValue,
	constant float* BNBetaValue,
	constant float* BNGammaValue,
	constant float* valueConnectionWeights,
	constant float* valueBiases,
	constant float* valueBiasLast,

// for policy layer
	constant float* convWeightsPolicy,
	constant float* BNMeansPolicy,
	constant float* BNStddevPolicy,
	constant float* BNBetaPolicy,
	constant float* BNGammaPolicy,
	constant float* policyConnectionWeights,
	constant float* policyBiases,
// for residual tower
	constant float* convFilterWeights,
// output
	global float* output,
// identify the run
	constant int* _networkIndex) // which NN weights to use from global memory
{
	private int globId = get_global_id(0);
	private int networkIndex = _networkIndex[globId];
	private int inputIndex = globId*50;
	private int outputIndex = globId*26;

	// local variables are shared by all work items of a work group
	// for now these are hardcoded here.. //

    private int filterWidth = 3;
    private int filterHeight = 3;
	private int gameboardWidth = 5;
	private int gameboardHeight = 5;
    private int nofPlanes = 2;
	private int nofOutputPolicies = 25; // policy net has 9 outputs (1 per potential move)
	private int nofOutputValues = 1; // value head has 1 output
	private int nofFilters = 6; //64- the convolution layer has 64 filters
	private int nofConvLayers = 7; // 13- currently 13 conv layers, 1 input, 2 in each of 6 residual layers
	private int nofResidualLayers = 3; // 6- half of (conv-1), 1 conv layer is for input (heads are seperate)
	private int nofPolicyPlanes = 4; // 32- for some reason we only want 32 planes in policy/value heads (the input to is 64 and
	private int nofValuePlanes = 4; //32- conv makes it 32) [cheat sheet alphazero go -> 2]
	private int valueHiddenLayerSize = 8; // was 128
	private float softmaxTemperature = 1.0f;

	// private array to work on, could re-use some later
    private float outputConvFilter[6 * 5 * 5]; // nofFilters *..
    private float outputResidualLayer[6 * 5 * 5]; // nofFilters *..
    private float temporary[6 * 5 * 5]; // nofFilters *..
    private float inputResidualLayer[6 * 5 * 5]; // nofFilters *..
	private float inputFCLayerPolicy[4 * 5* 5]; // nofpolicyplanes * ..
	private float outputValueData[4 * 5 * 5]; // nofvalueplanes* ..
	private float outputPolicyData[5 * 5]; // nofvalueplanes* ..
	private float localInput[5*5*2]; 
	private float temporaryValueData[8]; // valueHiddenLayerSize

    private float softmaxPolicy[25]; // nofOutputPolicies
    private float winrateOut[1];

	// copy input to inputreslayer because of access specifiers which may not change and because the input output is 
	// swapped to conv function call we also cant change the specifier in the argument, plus should be faster anyway
	for(int i = inputIndex; i < inputIndex + 50; ++i) {
		localInput[i] = input[i];	
	}

	////////////////////////////////////////////// start of network eval //////////////////////////////////////////////
	
	/*Conv layer */
	private int networkOffsetConv = networkIndex*nofFilters* nofPlanes * filterHeight * filterWidth; 
    Convolution(6 * 5 * 5, localInput, outputConvFilter, firstConvFilterWeights, nofPlanes, nofFilters, filterWidth, filterHeight, 0, networkOffsetConv);

	private int networkOffsetBN = networkIndex*nofConvLayers * nofFilters; 
    BN(6 * 5 * 5, outputConvFilter, inputResidualLayer, BNMeans, BNStddev, nofFilters, 0, BNGammas, BNBetas, networkOffsetBN);

    /*Residual tower*/
	networkOffsetConv = networkIndex * (2 * nofResidualLayers) * nofFilters * nofFilters * filterHeight * filterWidth;
    for (int index = 0; index < nofResidualLayers; index += 1) 
	{
        Convolution(6 * 5 * 5, inputResidualLayer, outputResidualLayer, convFilterWeights, nofFilters, nofFilters, filterWidth, filterHeight, index*2, networkOffsetConv);
        BN(6 * 5 * 5, outputResidualLayer, outputResidualLayer, BNMeans, BNStddev, nofFilters, index*2+1, BNGammas, BNBetas, networkIndex);
        Convolution(6 * 5 * 5, outputResidualLayer, temporary, convFilterWeights, nofFilters, nofFilters, filterWidth, filterHeight, index*2+1, networkOffsetConv);
        BNWithResidual(6 * 5 * 5, temporary, outputResidualLayer, inputResidualLayer, BNMeans, BNStddev, nofFilters, index*2+2, BNGammas, BNBetas, networkIndex);
                
        // temporary holds result
		for(int z = 0; z < 6 * 5 * 5; ++z) {
			inputResidualLayer[z] = outputResidualLayer[z];
		}
    }

    /*value head*/
	networkOffsetBN = networkIndex*nofValuePlanes; 
	networkOffsetConv = networkIndex * nofFilters* nofValuePlanes;

    Convolution(4 * 5 * 5, inputResidualLayer, outputValueData, convWeightsValue1, nofFilters, nofValuePlanes, 1, 1, 0, networkOffsetConv);
    BN(4 * 5 * 5, outputValueData, outputValueData, BNMeansValue, BNStddevValue, nofValuePlanes, 0, BNGammaValue, BNBetaValue, networkOffsetBN);

	private int networkOffsetFCLayer = networkIndex * gameboardHeight * gameboardWidth*nofValuePlanes * valueHiddenLayerSize;
	private int networkOffsetFCLayer2 = networkIndex * valueHiddenLayerSize;

    FCLayer(4 * 5 * 5, 8, outputValueData, temporaryValueData, valueConnectionWeights, valueBiases,  true, networkOffsetFCLayer, networkOffsetFCLayer2); // with rectifier

	networkOffsetFCLayer = networkIndex * valueHiddenLayerSize;
	networkOffsetFCLayer2 = networkIndex;

    FCLayer(8, 1, temporaryValueData, winrateOut, convWeightsValue2, valueBiasLast, false, networkOffsetFCLayer, networkOffsetFCLayer2); // 1 output, 1 bias
    float winrateSig = (1.0f + tanh(winrateOut[0])) / 2.0f;

    /*policy head*/
	networkOffsetBN = networkIndex*nofPolicyPlanes; 
	networkOffsetConv = networkIndex * nofPolicyPlanes* nofFilters;
    Convolution(4 * 5* 5, inputResidualLayer, inputFCLayerPolicy, convWeightsPolicy, nofFilters, nofPolicyPlanes, 1, 1, 0, networkOffsetConv);
    BN(4*5*5, inputFCLayerPolicy, inputFCLayerPolicy, BNMeansPolicy, BNStddevPolicy, nofPolicyPlanes, 0, BNGammaPolicy, BNBetaPolicy, networkOffsetBN);

	networkOffsetFCLayer = networkIndex * gameboardHeight * gameboardWidth* nofPolicyPlanes * nofOutputPolicies;
	networkOffsetFCLayer2 = networkIndex*nofOutputPolicies;

    FCLayer(4*5*5, 5*5, inputFCLayerPolicy, outputPolicyData, policyConnectionWeights, policyBiases, false, networkOffsetFCLayer, networkOffsetFCLayer2); // without rectifier
    Softmax(outputPolicyData, softmaxPolicy, softmaxTemperature);

	////////////////////////////////////////////// end of network eval //////////////////////////////////////////////

	for(int i = 0; i < 25; ++i) 
	{
		output[outputIndex+i] = softmaxPolicy[i];
	}
	output[outputIndex+25] = winrateSig;	
}

/*	See http://developer.amd.com/wordpress/media/2013/07/AMD_Accelerated_Parallel_Processing_OpenCL_Programming_Guide-rev-2.7.pdf
		https://cims.nyu.edu/~schlacht/OpenCLModel.pdf
	And also: https://stackoverflow.com/questions/26804153/opencl-work-group-concept
	regarding work groups and work-items
	private		Specific to a work-item; it is not visible to other work-items.
	local		Specific to a work-group; accessible only by work-items belonging to that
				work-group.
	global		Accessible to all work-items executing in a context, as well as to the host
				(read, write, and map commands).
	constant	Read-only region for host-allocated and -initialized objects that are not
				changed during kernel execution.
	host (CPU)	Host-accessible region for an application’s data structures and program
	data.
	PCIe		Part of host (CPU) memory accessible from, and modifiable by, the host
				program and the GPU compute device. Modifying this memory require

	TODO: copying data to local memory (or even private) could speed up the kernel
*/