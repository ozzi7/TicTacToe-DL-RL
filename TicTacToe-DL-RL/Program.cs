using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
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
            // ---------------------- NEUROEVOLUTION LOOP ----------------------
            //NeuralNetwork NN = new NeuralNetwork();
            //NN.InitializeWeights(0.1f);
            ////////NN.InitializeWeightsTo0();
            //NN.CalculateVirtualBNs();
            //////NN.ReadWeightsFromFile("weights_net_14_longgpurun.txt");
            //NN.SaveWeightsToFile("weights_neuro.txt");
            //NN.ReadWeightsFromFile("weights_neuro.txt");
            //Trainer trainer = new Trainer(NN);
            //trainer.Train();

            // -------------------------- KERAS BP LOOP ---------------------------
            NeuralNetwork NN = new NeuralNetwork();

            NN.ReadWeightsFromFileKeras("./../../../Training/weights.txt"); // must have been created with python script
            Trainer trainer = new Trainer(NN);
            trainer.TrainKeras();
        }
    }
}
