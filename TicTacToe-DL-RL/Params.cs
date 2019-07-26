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
        public static bool FORCE_KERNEL_RECOMPILE = false; // if false the kernel binary is loaded from file
        //public static int GPU_THREADS_AND_QUEUES = 2;
        public static int MAX_PARALLEL_KERNEL_EXECUTIONS = 8000; // opencl calls at most MAX_PARALLEL_KERNEL_EXECUTIONS and less if not enough data arrived from CPU //2304

        // set this to a even number, this also increases the quality of the tree search with higher threads because there are less virtual losses
        public static int NOF_CPU_THREADS_GPU_WORKLOAD = 512; // increases also the number of GPU memory used, if GPU used => one extra thread for openCL

        public static int NOF_CPU_THREADS_CPU_WORKLOAD = 4;
        public static int MAX_PENDING_NN_EVALS = 20; // should be lower than sims per move 
        // = how many NN evals are queued up in the MCTS tree before the CPU thread must wait for results
        // the MCTS search becomes less useful if it continues with fake data while waiting for the real outputs
        // it is better to keep this low and increase parallel trees (increasing number of CPU threads)
        // TODO: currently before a new move starts the CPU threads wait for all results first 

        public static float EPS = 0.001f; // for numerical stability in square roots etc.

        // NEUROEVOLUTION
        public static int NOF_OFFSPRING = 20; // must be 2n because half of NOF_OFFSPRING share same weight mutation but in opposite direction
        public static int NOF_GAMES_PER_OFFSPRING = 20;
        public static float NOISE_SIGMA = 0.03f;  // noise standard deviation 0.1 (default), 0.01 ok
        public static float LEARNING_RATE = 0.001f;
        public static int SHOW_SAMPLE_MATCHES_EVERY_XTH_EPOCH = 20;
        public static float WEIGHT_DECAY_FACTOR = 0.995f;
        public static int SAVE_WEIGHT_EVERY_XTH_EPOCH = 20;

        // BP only
        public static int NOF_GAMES_TRAIN_KERAS = 2048; // multiple of threads
        public static int STOCHASTIC_MOVES_FIRST_X_MOVES_TRAINING = 8;
        public static int STOCHASTIC_MOVES_FIRST_X_MOVES_TESTING = 4;

        // NEUROEVOLUTION + BP PARAMS
        public static int NOF_EPOCHS = 10000000;
        public static int NOF_GAMES_TEST = 64; // must be 2n for equal tests of player X and player Z
        public static int NOF_GAMES_VS_RANDOM = 16;
        public static int NOF_SIMS_PER_MOVE_TRAINING = 400;
        public static int NOF_SIMS_PER_MOVE_TESTING = 400;
        public static int NOF_SIMS_PER_MOVE_VS_RANDOM1 = 80;
        public static int NOF_SIMS_PER_MOVE_VS_RANDOM2 = 10;
        public static int NOF_SIMS_PER_MOVE_VS_RANDOM3 = 1;
        public static float C_PUCT = 1.0f;

        public static float MINIMUM_WIN_PERCENTAGE = 52.0f; // new networks must win at least x percent against old

        public static float DIRICHLET_NOISE_WEIGHT;
        public static bool USE_REAL_TERMINAL_VALUES = true;
        public static String PLOT_FILENAME = "plotdata.txt";

        // GAME SPECIFIC 
        public static int MAXIMUM_PLYS = 100; // for games which take too long to cut them off, tictactoe is done after <= 25 moves always 
        public static int boardSizeX = 5;
        public static int boardSizeY = 5;
    }
}
