using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    public static class Params
    { 
        public static bool GPU_ENABLED = true;
        public static int nofEpochs = 10000000;
        public static float c_puct = 1.41f; // roughly sqrt 2 // was 2
        public static int nofSimsPerPosTest = 20; // could/should be time
        public static int nofSimsPerPosTrain = 20; // could/should be time
        public static int populationSize = 20;

        public static int nofTrainingGames = 1000000;
        public static int nofTestGames = 60; // only 2 makes sense without noise
        public static int maxPlies = 100;

        public static int gamesPerIndividuum = 40;
        public static float sigma = 0.1f;  // noise standard deviation 0.1, 0.01, 2 sims, 1.4puct
        public static float alpha = 0.001f;// learning rate

        public static int boardSizeX = 5;
        public static int boardSizeY = 5;
        public static int rootChildren = 25; // for dirichlet noise
        public static float noiseWeight;
        public static float weightDecayFactor = 0.98f;

        public static int SaveWeightsEveryXthTrainingRun = 20;
    }
}
