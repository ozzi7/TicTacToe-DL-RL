#define BOARD_X 5
#define BOARD_Y 5
#define BOARD_SIZE 25
#define INPUT_PLANES 3

#define FILTER_HEIGHT 3
#define FILTER_WIDTH 3
#define RES_LAYERS 6
#define CONV_LAYERS 13
#define RES_LAYER_FILTERES 32
#define VALUE_FILTERS 32
#define POLICY_FILTERS 32
#define VALUE_HEAD_HIDDEN_LAYER_SIZE 32
#define VALUE_HEAD_OUTPUTS 1
#define POLICY_HEAD_OUTPUTS 25

static inline void Convolution(float* input, float* output, constant float* convWeights,
    int nofInputPlanes, int nofFilters, int filterWidth, int filterHeight, int index, int networkOffset)
{
    // zero padding

	for (int u = 0; u < nofFilters*BOARD_SIZE; ++u)
    {
        output[u] = 0.0f;
    }
    
	for (int i = 0; i < nofFilters; ++i) 
	{ 
		for (int j = 0; j < nofInputPlanes; ++j)
		{
			for (int k = 0; k < BOARD_Y; ++k)
			{
				for (int l = 0; l < BOARD_X; ++l)
				{
					// looking at a 1x1x1 of the input here, we sum up the 3x3 neighbors (depending on filter size)
					for (int x = 0; x < filterHeight; ++x)
					{
						for (int y = 0; y < filterWidth; ++y)
						{

							// going through the neighbors
							if (k - filterHeight / 2 + x < 0 || k - filterHeight / 2 + x >= BOARD_Y ||
								l - filterWidth / 2 + y < 0 || l - filterWidth / 2 + y >= BOARD_X)
							{
								// the filter is out of bounds, set to 0 (0 padding)
								continue;
							}
							// when input value is 0 skip all filters
							//if (input[j * 5 * 5 + k * 5 + l + (x - (filterHeight / 2)) * 5 + y - (filterWidth / 2)] == 0.0f)
							//{
							//    break;
							//}
							output[i * BOARD_Y * BOARD_X + k * BOARD_X + l] += 
								input[j * BOARD_Y * BOARD_X + k * BOARD_X + l + (x - (filterHeight / 2))*BOARD_X + y - (filterWidth/2)] *
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
}
static inline void BN(float* input, float* output, constant float* BNMeans, constant float* BNStdDev, 
	int nofFilters, int index, constant float* BNGammas, constant float* BNBetas, int networkIndex)
{
    // without residual add
    for (int i = 0; i < nofFilters; ++i)
    {
        // go through each plane coming into BN and apply to each element the means and stddev..
        for (int j = 0; j < BOARD_SIZE; ++j)
        {
            // we know the size of one plane by dividing input through number of planes (input.length/nofFilters)
            // batch norm/ batch stddev
            /* see Alg 1: https://arxiv.org/pdf/1502.03167.pdf */
            float x_til = (float)((input[i * BOARD_SIZE + j] - BNMeans[networkIndex +  index * nofFilters + i])/
                (BNStdDev[networkIndex +  index * nofFilters + i]));
            output[i * BOARD_SIZE + j] = BNGammas[networkIndex + index * nofFilters + i] *
					x_til+BNBetas[networkIndex + index * nofFilters + i];

            // relu
            if (output[i * BOARD_SIZE + j] < 0.0f)
                output[i* BOARD_SIZE + j] = 0.3f*output[i* BOARD_SIZE + j];
        }
    }
}
static inline void BNWithResidual(float* input, float* output, float* residual, constant float* BNMeans, constant float* BNStdDev, 
	int nofFilters, int index, constant float* BNGammas, constant float* BNBetas, int networkIndex)
{
    for (int i = 0; i < nofFilters; ++i)
    {
        for (int j = 0; j < BOARD_SIZE; ++j)
        {
            // batch norm/ batch stddev
            float x_til = (float)((input[i *BOARD_SIZE + j] - BNMeans[networkIndex + index * nofFilters + i]) /
                (BNStdDev[networkIndex +index * nofFilters + i]));

            output[i * BOARD_SIZE + j] = residual[i * BOARD_SIZE + j] + BNGammas[networkIndex + index * nofFilters + i] * x_til + 
																		BNBetas[networkIndex + index * nofFilters + i];

            // relu
            if (output[i * BOARD_SIZE + j] < 0.0f)
                output[i * BOARD_SIZE + j] = 0.3f*output[i *BOARD_SIZE + j] ;
        }
    }
}
static inline void FCLayer(int sizeofInput, int sizeofOutput, float* input, float* output, constant float* connectionWeights, constant float* outputBiases, bool rectifier, 
int connectionWeightIndex, int outputBiasesIndex)
{
    for (int i = 0; i < sizeofOutput; ++i)
    {
		output[i] = 0.0f;
        for (int j = 0; j < sizeofInput; ++j)
        {
            output[i] += input[j] * connectionWeights[connectionWeightIndex + j*sizeofOutput+i];
        }
        output[i] += outputBiases[outputBiasesIndex + i];

        if(rectifier && output[i] < 0.0f)
        {
            output[i] = 0.3f*output[i];
        }
    }
}
static inline void Softmax(float* input, float* output, float temperature)
{
    // must be input length == output length
	float alpha = -FLT_MAX;	
	for(int i = 0; i < POLICY_HEAD_OUTPUTS; ++i) {
		if(input[i] >= alpha) {
			alpha = input[i];
		}
	}
    alpha /= temperature;

    float denom = 0.0f;
    float helper[POLICY_HEAD_OUTPUTS];
    for (int i = 0; i < POLICY_HEAD_OUTPUTS; i++) {
        float val = (float)exp((input[i] / temperature) - alpha);
        helper[i] = val;
        denom += val;
    }

    for (int i = 0; i < POLICY_HEAD_OUTPUTS; ++i)
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
            data[i] = 0.3f*data[i];
    }
}

/* ENTRY_POINT */
kernel void NN(
// for input layer
	constant float* restrict input,
	constant float* restrict firstConvFilterWeights,
// shared among all layers
	constant float* restrict BNMeans,
	constant float* restrict BNStddev,
	constant float* restrict BNBetas,
	constant float* restrict BNGammas,
// for value layer
	constant float* restrict convWeightsValue1,
	constant float* restrict valueConnectionWeights2,

	constant float* restrict BNMeansValue,
	constant float* restrict BNStddevValue,
	constant float* restrict BNBetaValue,
	constant float* restrict BNGammaValue,

	constant float* restrict valueConnectionWeights,
	constant float* restrict valueBiases,
	constant float* restrict valueBiasLast,

// for policy layer
	constant float* restrict convWeightsPolicy,

	constant float* restrict BNMeansPolicy,
	constant float* restrict BNStddevPolicy,
	constant float* restrict BNBetaPolicy,
	constant float* restrict BNGammaPolicy,

	constant float* restrict policyConnectionWeights,
	constant float* restrict policyBiases,
// for residual tower
	constant float* restrict convFilterWeights,
// output
	global float* restrict output,
 // which NN weights to use from global memory
	constant int* restrict _networkIndex)
{
	private int globId = get_global_id(0);
	private int networkIndex = _networkIndex[globId]; //
	private int inputIndex = globId*INPUT_PLANES*BOARD_SIZE;
	private int outputIndex = globId*(POLICY_HEAD_OUTPUTS+VALUE_HEAD_OUTPUTS);

	private float softmaxTemperature = 1.0f;

	// private array to work on, could re-use some later
    private float outputConvFilter[RES_LAYER_FILTERES *BOARD_SIZE]; 
    private float outputResidualLayer[RES_LAYER_FILTERES *BOARD_SIZE];
    private float temporary[RES_LAYER_FILTERES *BOARD_SIZE];
    private float inputResidualLayer[RES_LAYER_FILTERES *BOARD_SIZE];
	private float inputFCLayerPolicy[POLICY_FILTERS *BOARD_SIZE];
	private float outputValueData[VALUE_FILTERS *BOARD_SIZE];
	private float outputPolicyData[POLICY_HEAD_OUTPUTS];
	private float localInput[INPUT_PLANES*BOARD_SIZE]; 
	private float temporaryValueData[VALUE_HEAD_HIDDEN_LAYER_SIZE];

    private float softmaxPolicy[POLICY_HEAD_OUTPUTS]; 
    private float winrateOut[VALUE_HEAD_OUTPUTS];

	// copy input to inputreslayer because of access specifiers which may not change and because the input output is 
	// swapped to conv function call we also cant change the specifier in the argument, plus should be faster anyway
	for(int i = 0; i < INPUT_PLANES*BOARD_SIZE; ++i) {
		localInput[i] = input[inputIndex + i];	
	}
	
	////////////////////////////////////////////// start of network eval //////////////////////////////////////////////
	
	/*Conv layer */
	private int networkOffsetConv = networkIndex*RES_LAYER_FILTERES* INPUT_PLANES * FILTER_HEIGHT * FILTER_WIDTH;

    Convolution(localInput, outputConvFilter, firstConvFilterWeights, INPUT_PLANES, RES_LAYER_FILTERES, FILTER_WIDTH, FILTER_HEIGHT, 0, networkOffsetConv);
	
	private int networkOffsetBN = networkIndex*CONV_LAYERS * RES_LAYER_FILTERES;  //correct
    BN(outputConvFilter, inputResidualLayer, BNMeans, BNStddev, RES_LAYER_FILTERES, 0, BNGammas, BNBetas, networkOffsetBN);

    /*Residual tower*/
	networkOffsetConv = networkIndex * (2 * RES_LAYERS) * RES_LAYER_FILTERES * RES_LAYER_FILTERES * FILTER_HEIGHT * FILTER_WIDTH; //correct
    for (int index = 0; index < RES_LAYERS; index += 1) 
	{
        Convolution(inputResidualLayer, outputResidualLayer, convFilterWeights, RES_LAYER_FILTERES, RES_LAYER_FILTERES, FILTER_WIDTH, FILTER_HEIGHT, index*2, networkOffsetConv);
        BN(outputResidualLayer, outputResidualLayer, BNMeans, BNStddev, RES_LAYER_FILTERES, index*2+1, BNGammas, BNBetas, networkOffsetBN);
        Convolution(outputResidualLayer, temporary, convFilterWeights, RES_LAYER_FILTERES, RES_LAYER_FILTERES, FILTER_WIDTH, FILTER_HEIGHT, index*2+1, networkOffsetConv);
        BNWithResidual(temporary, outputResidualLayer, inputResidualLayer, BNMeans, BNStddev, RES_LAYER_FILTERES, index*2+2, BNGammas, BNBetas, networkOffsetBN);
                
        // temporary holds result
		for(int z = 0; z < RES_LAYER_FILTERES *BOARD_SIZE; ++z) {
			inputResidualLayer[z] = outputResidualLayer[z];
		}
    }

    /*value head*/
	networkOffsetBN = networkIndex*VALUE_FILTERS;
	networkOffsetConv = networkIndex * RES_LAYER_FILTERES* VALUE_FILTERS*1*1;

    Convolution(inputResidualLayer, outputValueData, convWeightsValue1, RES_LAYER_FILTERES, VALUE_FILTERS, 1, 1, 0, networkOffsetConv);
    BN(outputValueData, outputValueData, BNMeansValue, BNStddevValue, VALUE_FILTERS, 0, BNGammaValue, BNBetaValue, networkOffsetBN);

	private int networkOffsetFCLayerBiases = networkIndex * VALUE_HEAD_HIDDEN_LAYER_SIZE;
	private int networkOffsetFCLayerWeights = networkIndex * BOARD_SIZE*VALUE_FILTERS * VALUE_HEAD_HIDDEN_LAYER_SIZE;

    FCLayer(VALUE_FILTERS *BOARD_SIZE, VALUE_HEAD_HIDDEN_LAYER_SIZE, outputValueData, temporaryValueData, valueConnectionWeights, valueBiases, true, networkOffsetFCLayerWeights, networkOffsetFCLayerBiases); // with rectifier

	networkOffsetFCLayerBiases = networkIndex;
	networkOffsetFCLayerWeights = networkIndex * VALUE_HEAD_HIDDEN_LAYER_SIZE;

    FCLayer(VALUE_HEAD_HIDDEN_LAYER_SIZE, 1, temporaryValueData, winrateOut, valueConnectionWeights2, valueBiasLast, false, networkOffsetFCLayerWeights, networkOffsetFCLayerBiases); // 1 output, 1 bias
    float winrateSig = tanh(winrateOut[0]);

    /*policy head*/
	networkOffsetBN = networkIndex*POLICY_FILTERS; 
	networkOffsetConv = networkIndex * POLICY_FILTERS* RES_LAYER_FILTERES;
    Convolution(inputResidualLayer, inputFCLayerPolicy, convWeightsPolicy, RES_LAYER_FILTERES, POLICY_FILTERS, 1, 1, 0, networkOffsetConv);
    BN(inputFCLayerPolicy, inputFCLayerPolicy, BNMeansPolicy, BNStddevPolicy, POLICY_FILTERS, 0, BNGammaPolicy, BNBetaPolicy, networkOffsetBN);

	networkOffsetFCLayerBiases = networkIndex*POLICY_HEAD_OUTPUTS; // correct
	networkOffsetFCLayerWeights = networkIndex * BOARD_SIZE * POLICY_FILTERS * POLICY_HEAD_OUTPUTS; // correct

    FCLayer(POLICY_FILTERS*BOARD_SIZE, POLICY_HEAD_OUTPUTS, inputFCLayerPolicy, outputPolicyData, policyConnectionWeights, policyBiases, false, networkOffsetFCLayerWeights, networkOffsetFCLayerBiases); // without rectifier
    Softmax(outputPolicyData, softmaxPolicy, softmaxTemperature);
	
	////////////////////////////////////////////// end of network eval //////////////////////////////////////////////

	for(int i = 0; i < POLICY_HEAD_OUTPUTS; ++i) 
	{
		output[outputIndex+i] = softmaxPolicy[i];
	}
	output[outputIndex+POLICY_HEAD_OUTPUTS] = winrateSig;	
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