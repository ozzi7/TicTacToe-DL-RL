using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    public static class Params
    {
        public static double learningRate = 0.001;
        public static double dropout = 0.3;
        public static int nofEpochs = 10;
        public static int batchSize = 64;
        public static int nofChannels = 512;

        public static int nofTrainingGames = 100;
        public static int maxPlies = 100;

        public static int boardSizeX = 3;
        public static int boardSizeY = 3;
    }
    public static class RandomNr
    {
        private static Random random = new Random();
        public static int GetInt(int from, int to)
        {
            return random.Next(from, to);
        }
    }
}
