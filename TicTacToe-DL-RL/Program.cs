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
            if (args.Count() >= 2 && args[0].Equals("play"))
            {
                PlayVsHuman(args);
            }
            else
            {
                Train();
            }
        }
        static void Train()
        {
            if (false)
            {
                // ---------------------- NEUROEVOLUTION LOOP ----------------------
                NeuralNetwork NN = new NeuralNetwork();
                NN.InitializeWeights(0.1f);
                //////NN.InitializeWeightsTo0();
                NN.CalculateVirtualBNs();
                ////NN.ReadWeightsFromFile("weights_net_14_longgpurun.txt");
                NN.SaveWeightsToFile("weights_neuro.txt");
                NN.ReadWeightsFromFile("weights_neuro.txt");
                Trainer trainer = new Trainer(NN);
                trainer.Train();
            }
            else
            {
                // -------------------------- KERAS BP LOOP ---------------------------
                NeuralNetwork NN = new NeuralNetwork();

                NN.ReadWeightsFromFileKeras("./../../../Training/weights.txt"); // must have been created with python script
                Trainer trainer = new Trainer(NN);
                //trainer.ValidateOuputGPU();
                trainer.TrainKeras();
            }
        }
        static void PlayVsHuman(string[] args)
        {
            Params.DIRICHLET_NOISE_WEIGHT = 0.0f;
            Params.USE_REAL_TERMINAL_VALUES = true;
            Params.GPU_ENABLED = false;

            NeuralNetwork nn = new NeuralNetwork();
            nn.ReadWeightsFromFileKeras("./../../../Training/" + args[1]);

            for (int i = 0; i < 100; ++i)
            {
                Game game = new Game();
                NNPlayer nnPlayer = new NNPlayer(nn);
                HumanPlayer humanPlayer = new HumanPlayer();

                Player evaluationNetworkPlayer = i % 2 == 0 ? Player.X : Player.Z;
                Console.WriteLine("New game!");

                for (int curr_ply = 0; curr_ply < GameProperties.MAXIMUM_PLYS; ++curr_ply)
                {
                    if (game.IsOver())
                    {
                        Console.WriteLine("Score: " + game.GetScore());
                        break;
                    }

                    Tuple<int, int> move;
                    if (game.position.sideToMove == evaluationNetworkPlayer)
                    {
                        Console.WriteLine("Calculating move...\n");
                        move = nnPlayer.GetMove(game, Params.NOF_SIMS_PER_MOVE_TESTING);
                        nnPlayer.mcts.PrintPolicy();
                    }
                    else
                    {
                        move = humanPlayer.GetMove(game);
                    }

                    game.DoMove(move);
                    nnPlayer.DoMove(move);

                    Console.WriteLine(game.position.ToString());
                }
            }
        }
    }
}
