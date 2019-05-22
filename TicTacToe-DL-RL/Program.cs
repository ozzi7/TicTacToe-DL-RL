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
            NeuralNetwork NN = new NeuralNetwork();
            NN.SaveWeightsToFile("weights_start.txt");
            NN.ReadWeightsFromFile("weights_start.txt");
            NN.CalculateVirtualBNs();
            
            //NeuralNetwork NN = new NeuralNetwork();
            //NN.ReadWeightsFromFile("weights_net_451.txt");

            //NeuralNetwork previousNN = new NeuralNetwork();
            //previousNN.ReadWeightsFromFile("weights_net_451.txt");


            //NeuralNetwork NN = new NeuralNetwork();
            //NeuralNetwork previousNN = new NeuralNetwork();
            Trainer trainer = new Trainer(NN);
            trainer.Train();
            //nn.SaveToFile("test.txt");
        }
    }
}
