using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    public static class Params
    {
        public static int MAX_KERNEL_EXECUTIONS = 40; // opencl waits for x inputs and then executes all kernels = pop size
        public static bool GPU_ENABLED = true;
        /* the number of neural networks to keep in GPU memory at the same time */
        public static int NOF_NNs = 80; // populationSize*2(currentNN)

        public static int nofEpochs = 10000000;
        public static float c_puct = 1.41f; // roughly sqrt 2 // was 2
        public static int nofSimsPerPosTest = 40; // could/should be time
        public static int nofSimsPerPosTrain = 40; // could/should be time
        public static int populationSize = 40; // must be 2n

        public static int nofTrainingGames = 1000000;
        public static int nofTestGames = 60; // only 2 makes sense without noise
        public static int maxPlies = 100;

        public static int gamesPerIndividuum = 80;
        public static float sigma = 0.1f;  // noise standard deviation 0.1, 0.01, 2 sims, 1.4puct
        public static float alpha = 0.001f;// learning rate

        public static int boardSizeX = 5;
        public static int boardSizeY = 5;
        public static int rootChildren = 25; // for dirichlet noise
        public static float noiseWeight;
        public static float weightDecayFactor = 0.98f;

        public static int SaveWeightsEveryXthTrainingRun = 20;

        public static int ID = 0;
        public static int getGlobalID()
        {
            return ID++;
        }
    }
}
