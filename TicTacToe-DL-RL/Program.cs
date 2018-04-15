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
            Train();
        }
        public void Train()
        {
            NeuralNetwork nn = new NeuralNetwork();
            Trainer trainer = new Trainer(nn);
            trainer.Train();
            nn.SaveToFile("test.txt");
        }
    }
}
