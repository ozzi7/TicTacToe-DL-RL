using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    class Program
    {
        static void Main(string[] args)
        {
            Train();
        }
        static void Train()
        {
            NeuralNetwork nn = new NeuralNetwork();
            Trainer trainer = new Trainer(nn);
            trainer.Train();
            nn.SaveToFile("test.txt");
        }
    }
}
