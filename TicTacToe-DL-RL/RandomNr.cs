using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    public static class RandomNr
    {
        private static Random random = new Random();
        public static int GetInt(int fromInclusive, int toExclusive)
        {
            return random.Next(fromInclusive, toExclusive);
        }
        public static float GetFloat(int from, int to)
        {
            return (float)(from + random.NextDouble() * (to - from));
        }
        public static float GetGaussianFloat()
        {
            double u1 = 1.0 - random.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return (float)randStdNormal;
        }
        public static float GetGaussianFloatCloseTo0()
        {
            double u1 = 1.0 - random.NextDouble(); //uniform(0,1] random doubles
            double u2 = 1.0 - random.NextDouble();
            double randStdNormal = Math.Sqrt(-2.0 * Math.Log(u1)) *
                         Math.Sin(2.0 * Math.PI * u2); //random normal(0,1)
            return (float)randStdNormal*0.1f;
        }
    }
}
