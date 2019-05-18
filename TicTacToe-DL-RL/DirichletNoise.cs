using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;

namespace TicTacToe_DL_RL
{
    public static class DirichletNoise
    {
        private static Dirichlet dirichlet;
        private static double[] dirichletNoise;
        public static void InitDirichlet(int nofChildren)
        {
            // assume 10 random games, 9 potential moves => 10/9
            float init = 10.0f / nofChildren;
            double[] alphaInit = new double[nofChildren];

            for (int i = 0; i < nofChildren; ++i)
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
            if (sumDirichlet != 0.0f)
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
