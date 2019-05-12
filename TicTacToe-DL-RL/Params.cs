using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;

namespace TicTacToe_DL_RL
{
    public static class Params
    { 
        public static int nofEpochs = 1000000;
        public static float c_puct = 1.41f; // roughly sqrt 2 // was 2
        public static int nofSimsPerPosTest = 10; // could/should be time
        public static int nofSimsPerPosTrain = 10; // could/should be time
        public static int populationSize = 10;

        public static int nofTrainingGames = 1000000;
        public static int nofTestGames = 20; // only 2 makes sense without noise
        public static int maxPlies = 100;

        public static int gamesPerIndividuum = 10;
        public static float sigma = 0.1f;  // noise standard deviation 0.1, 0.01, 2 sims, 1.4puct
        public static float alpha = 0.1f;// learning rate

        public static int boardSizeX = 5;
        public static int boardSizeY = 5;
        public static int rootChildren = 25; // for dirichlet noise
        public static float noiseWeight;
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
            return (float)(from + random.NextDouble() * (to-from));
        }
        public static float GetGaussianFloat()
        {
            double u1 = 1.0 - random.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return (float)randStdNormal;
        }
    }
    public static class DirichletNoise {
        private static Dirichlet dirichlet;
        private static double[] dirichletNoise;
        public static void InitDirichlet(int nofChildren)
        {
            // assume 10 random games, 9 potential moves => 10/9
            float init = 10.0f / nofChildren;
            double[] alphaInit = new double[nofChildren];

            for(int i = 0; i < nofChildren; ++i)
            {
                alphaInit[i] = init;
            }

            dirichlet = new Dirichlet(alphaInit); 
            dirichletNoise = dirichlet.Sample();

            float sumDirichlet = 0.0f;
            for (int i = 0; i < nofChildren; ++i)
            {
                sumDirichlet += (float)dirichletNoise[i];
            }
            if(sumDirichlet != 0.0f)
            {
                for (int i = 0; i < nofChildren; ++i)
                {
                    dirichletNoise[i] /= sumDirichlet;
                }
            }
        }
        public static float GetNoise(int index)
        {
            return (float)dirichletNoise[index];
        }
    }
}
