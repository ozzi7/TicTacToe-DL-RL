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
        public static bool GPU_ENABLED = false;
        public static int MAX_PARALLEL_KERNEL_EXECUTIONS = 2304; // opencl calls at most MAX_PARALLEL_KERNEL_EXECUTIONS and less if not enough data arrived from CPU

        public static int MAX_THREADS_CPU = 64; // increases also the number of GPU memory used, if GPU used => one extra thread for openCL max 64
        public static int MAX_PENDING_NN_EVALS = 40; // should be lower than sims per move 
        // = how many NN evals are queued up in the MCTS tree before the CPU thread must wait for results
        // the MCTS search becomes less useful if it continues with fake data while waiting for the real outputs
        // it is better to keep this low and increase parallel trees (increasing number of CPU threads)
        // TODO: currently before a new move starts the CPU threads wait for all results first 

        // NEUROEVOLUTION PARAMS
        public static int NOF_EPOCHS = 10000000;
        public static int NOF_OFFSPRING = 16; // must be 2n because half of NOF_OFFSPRING share same weight mutation but in opposite direction
        public static int NOF_GAMES_PER_OFFSPRING = 16;
        public static int NOF_GAMES_TEST = 16; // must be 2n for equal tests of player X and player Z
        public static int NOF_GAMES_VS_RANDOM = 2;
        public static int NOF_SIMS_PER_MOVE_TRAINING = 4; // could/should be time
        public static int NOF_SIMS_PER_MOVE_TESTING = 4; // could/should be time

        public static float C_PUCT = 4.0f; // in theory sqrt(2), in practice usually higher (=more exploration) for training
        public static float NOISE_SIGMA = 0.1f;  // noise standard deviation 0.1 (default), 0.01 ok
        public static float LEARNING_RATE = 0.0005f;
        public static float WEIGHT_DECAY_FACTOR = 0.99f;
        public static float DIRICHLET_NOISE_WEIGHT;
        public static DIRICHLET_NOISE_SCALING DN_SCALING = DIRICHLET_NOISE_SCALING.CONSTANT; // as a function of depth in mcts search tree
        public static int SHOW_SAMPLE_MATCHES_EVERY_XTH_EPOCH = 20;
        public static int SAVE_WEIGHT_EVERY_XTH_EPOCH = 20;
        public static String PLOT_FILENAME = "plotdata.txt";

        // GAME SPECIFIC 
        public static int MAXIMUM_PLYS = 100; // when to stop playing a game completely and declare draw (in tic tac toe game is always finished in 100 moves)
        public static int boardSizeX = 5;
        public static int boardSizeY = 5;
    }
    public enum DIRICHLET_NOISE_SCALING { CONSTANT, LINEAR, QUADRATIC };
}
