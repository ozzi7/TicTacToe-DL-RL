using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    public static class Params
    {
        public static float learningRate = 0.001f;
        public static float dropout = 0.3f;
        public static int nofEpochs = 10;
        public static int batchSize = 64;
        public static int nofChannels = 512;
        public static float c_puct = 0.3f; // TODO:
        public static int nofSimsPerPos = 11; // could/should be time

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
        public static float GetFloat(int from, int to)
        {
            return (float)(from + random.NextDouble() * (from-to));
        }
    }
}
