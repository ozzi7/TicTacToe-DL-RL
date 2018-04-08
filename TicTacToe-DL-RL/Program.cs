using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    class Program
    {
        void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork();

            Train();
        }
        void Train()
        {
            Trainer trainer = new Trainer();
        }
    }
}
