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

            while (true)
            {
                Trainer trainer = new Trainer(NN);

                String filename = trainer.ProduceTrainingGamesKeras(200);

                //trainer.Train();
                ProcessStartInfo pythonInfo = new ProcessStartInfo();
                Process python;
                pythonInfo.FileName = @"python.exe";//@"C:\Users\Admin\Anaconda3\envs\NALU\python.exe";
                pythonInfo.Arguments = "\"Z:\\CloudStation\\GitHub Projects\\TicTacToe-DL-RL\\Training\\main.py \" " + filename; // TODO: should be relative
                pythonInfo.CreateNoWindow = false;
                pythonInfo.UseShellExecute = false;

                var location = new Uri(Assembly.GetEntryAssembly().GetName().CodeBase);
                String exePath = new FileInfo(location.AbsolutePath).Directory.FullName;

                pythonInfo.RedirectStandardOutput = true;

                Console.WriteLine("Python Starting");
                python = Process.Start(pythonInfo);

                trainer.CheckPerformanceVsRandomKeras(20);

                while (!python.StandardOutput.EndOfStream)
                {
                    string line = python.StandardOutput.ReadLine();
                    Console.WriteLine(line);
                }

                python.WaitForExit();
                python.Close();

                NN.ReadWeightsFromFileKeras("./../../../Training/weights.txt");
            }
            //trainer.ValidateOuput();
            //Console.WriteLine("Done");
            //Console.ReadLine();
        }
    }
}
