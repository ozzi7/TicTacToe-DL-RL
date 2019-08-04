using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Distributions;

namespace TicTacToe_DL_RL
{
    public class DirichletNoise
    {
        private Dirichlet dirichlet;
        private double[] dirichletNoise;
        public DirichletNoise(int nofChildren)
        {
            // assume 10 random sims, 25 potential moves => 10/25
            //float init = 10.0f / nofChildren;
            float init = 10.0f / GameProperties.MOVES_IN_START_POS; // small number => sharper distribution
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
        public float GetNoise(int index)
        {
            return (float)dirichletNoise[index];
        }
    }
}
