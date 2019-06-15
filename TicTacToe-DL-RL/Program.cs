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

            /* to create a new network */
            //NN.InitializeWeights(1.0f);
            //NN.InitializeWeightsTo0();
            //NN.CalculateVirtualBNs();
            //NN.SaveWeightsToFile("weights_start.txt");
            //NN.ReadWeightsFromFile("weights_start.txt");


            /* to load from file */
            //NN.ReadWeightsFromFile("weights_net_14_longgpurun.txt");
            NN.ReadWeightsFromFileKeras("./../../../Training/weights.txt"); 
            Trainer trainer = new Trainer(NN);
            //trainer.Train();
            ////trainer.CheckPerformanceVsRandomKeras(20);
            //trainer.ProduceTrainingGamesKeras(300);
            trainer.ValidateOuput();
            Console.WriteLine("Done");
            Console.ReadLine();
        }
    }
}
