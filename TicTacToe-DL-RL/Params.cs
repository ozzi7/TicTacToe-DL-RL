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
        //public static int GPU_THREADS_AND_QUEUES = 2;
        public static int MAX_PARALLEL_KERNEL_EXECUTIONS = 1000; // opencl calls at most MAX_PARALLEL_KERNEL_EXECUTIONS and less if not enough data arrived from CPU //2304

        public static int MAX_THREADS_CPU = 4; // increases also the number of GPU memory used, if GPU used => one extra thread for openCL max 64
        public static int MAX_PENDING_NN_EVALS = 4; // should be lower than sims per move 
        // = how many NN evals are queued up in the MCTS tree before the CPU thread must wait for results
        // the MCTS search becomes less useful if it continues with fake data while waiting for the real outputs
        // it is better to keep this low and increase parallel trees (increasing number of CPU threads)
        // TODO: currently before a new move starts the CPU threads wait for all results first 

        public static float EPS = 0.001f; // for numerical stability in square roots etc.

        // NEUROEVOLUTION + BP PARAMS
        public static int NOF_EPOCHS = 10000000;
        public static int NOF_OFFSPRING = 20; // must be 2n because half of NOF_OFFSPRING share same weight mutation but in opposite direction
        public static int NOF_GAMES_PER_OFFSPRING = 10;
        public static int NOF_GAMES_TEST = 24; // must be 2n for equal tests of player X and player Z
        public static int NOF_GAMES_VS_RANDOM = 10;
        public static int NOF_SIMS_PER_MOVE_TRAINING = 20; // could/should be time
        public static int NOF_SIMS_PER_MOVE_TESTING = 20; // could/should be time
        public static int NOF_SIMS_PER_MOVE_VS_RANDOM1 = 80;
        public static int NOF_SIMS_PER_MOVE_VS_RANDOM2 = 10;
        public static int NOF_SIMS_PER_MOVE_VS_RANDOM3 = 1;
        public static float PERCENT_GROUND_TRUTH = 100.0f;
        public static float C_PUCT = 3.5f; // in theory sqrt(2), in practice usually higher (=more exploration) for training
        public static float NOISE_SIGMA = 0.03f;  // noise standard deviation 0.1 (default), 0.01 ok
        public static float LEARNING_RATE = 0.001f;
        public static float MINIMUM_WIN_PERCENTAGE = 53.0f; // new networks must win at least x percent against old
        public static float WEIGHT_DECAY_FACTOR = 0.995f;
        public static float DIRICHLET_NOISE_WEIGHT;
        public static DIRICHLET_NOISE_SCALING DN_SCALING = DIRICHLET_NOISE_SCALING.FIRST_NODE_ONLY; // as a function of depth in mcts search tree
        public static int SHOW_SAMPLE_MATCHES_EVERY_XTH_EPOCH = 20;
        public static int SAVE_WEIGHT_EVERY_XTH_EPOCH = 20;
        public static String PLOT_FILENAME = "plotdata.txt";
        public static float FPU_VALUE = 0.2f; // first play urgency, between [0,1], added to the puct score of unvisited children, this is needed if policy is 0 otherwise the children cannot be visited

        // BP only
        public static int NOF_GAMES_TRAIN_KERAS = 100;

        // GAME SPECIFIC 
        public static int MAXIMUM_PLYS = 100; // when to stop playing a game completely and declare draw (in tic tac toe game is always finished in 100 moves)
        public static int boardSizeX = 5;
        public static int boardSizeY = 5;
    }
    public enum DIRICHLET_NOISE_SCALING { CONSTANT, LINEAR, QUADRATIC, FIRST_NODE_ONLY };
}
