using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    public static class Params
    {
        // HARDWARE SETTINGS
        public static bool GPU_ENABLED = true;
        public static int MAX_KERNEL_EXECUTIONS = 12; // opencl waits for x inputs and then executes all kernels

        public static int MAX_THREADS_CPU = 4;

        // NEUROEVOLUTION PARAMS
        public static int NOF_EPOCHS = 10000000;
        public static int NOF_OFFSPRING = 12; // must be 2n because half of popsize share same weight mutation but in opposite direction
        public static int NOF_GAMES_PER_OFFSPRING = 16;
        public static int NOF_GAMES_TEST = 40; // must be 2n for equal tests of player X and player Z
        public static int NOF_GAMES_VS_RANDOM = 40;
        public static int NOF_SIMS_PER_MOVE_TRAINING = 2; // could/should be time
        public static int NOF_SIMS_PER_MOVE_TESTING = 2; // could/should be time

        public static float C_PUCT = 2.0f; // roughly sqrt 2 // was 2
        public static float NOISE_SIGMA = 0.1f;  // noise standard deviation 0.1 (default), 0.01 ok
        public static float LEARNING_RATE = 0.01f;
        public static float WEIGHT_DECAY_FACTOR = 0.99f;

        public static int SAVE_WEIGHT_EVERY_Xth_EPOCH = 20;
        public static String PLOT_FILENAME = "plotdata.txt";
        public static float DIRICHLET_NOISE_WEIGHT;

        // GAME SPECIFIC 
        public static int MAXIMUM_PLYS = 100; // when to stop playing a game completely and declare draw (in tic tac toe game is always finished in 100 moves)
        public static int boardSizeX = 5;
        public static int boardSizeY = 5;
    }
}
