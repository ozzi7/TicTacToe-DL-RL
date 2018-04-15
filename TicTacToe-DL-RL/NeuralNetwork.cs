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
        const int nofPlanes = 2; // input channels, board 9x9 + color 9x9
        const int nofOutputPolicies = 9; // policy net has 9 outputs (1 per potential move)
        const int nofOutputValues = 1; // value head has 1 output
        const int nofTiles = width * height / 3; // nof convolution "units"

        public NeuralNetwork()
        {
            
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
            List<float> output = Convolution(inputData);
        }
        public void SaveToFile(string filename)
        {

        }
        public List<float> Convolution(List<float> inputData)
        {
            // convolution on width*height volume
            List<float> result = new List<float>();
            return result;
        }
    }
}
