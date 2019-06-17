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
            //NN.InitializeWeights(1.0f);
            ////NN.InitializeWeightsTo0();
            //NN.CalculateVirtualBNs();
            //NN.SaveWeightsToFile("weights_start.txt");
            //NN.ReadWeightsFromFile("weights_start.txt");
            //Trainer trainer = new Trainer(NN);
            //trainer.Train();

            //NN.ReadWeightsFromFile("weights_net_14_longgpurun.txt");

            // -------------------------- KERAS BP LOOP ---------------------------
            NeuralNetwork NN = new NeuralNetwork();

            NN.ReadWeightsFromFileKeras("./../../../Training/weights.txt"); // must have been created with python script
            Trainer trainer = new Trainer(NN);
            trainer.TrainKeras();
        }
    }
}
