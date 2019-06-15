﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Threading;
using System.Diagnostics;

namespace TicTacToe_DL_RL
{
    class Trainer
    {
        private NeuralNetwork currentNN;
        private NeuralNetwork bestNN;

        private MovingAverage winsAsXMovingAvg = new MovingAverage();
        private MovingAverage winsAsZMovingAvg = new MovingAverage();
        private MovingAverage drawsMovingAvg = new MovingAverage();
        private MovingAverage averageMovesMovingAvg = new MovingAverage();
        private MovingAverage winrateVsRandMovingAvg1 = new MovingAverage();
        private MovingAverage winrateVsRandMovingAvg2 = new MovingAverage();
        private MovingAverage winrateVsRandMovingAvg3 = new MovingAverage();

        private double currentPseudoELO = 0;
        private float winrateVsRandTotal1 = -1;
        private float winrateVsRandTotal2 = -1;
        private float winrateVsRandTotal3 = -1;

        public Trainer(NeuralNetwork aCurrentNN)
        {
            currentNN = aCurrentNN;

            if (File.Exists(Params.PLOT_FILENAME))
                File.Delete(Params.PLOT_FILENAME);
            
            if (Params.GPU_ENABLED)
                OpenCL.Init(Math.Max(Params.NOF_OFFSPRING*2, Params.NOF_GAMES_TEST*2));
        }

        public void Train()
        {
            bestNN = new NeuralNetwork(currentNN.weights, currentNN.untrainable_weights);

            currentNN.SaveWeightsToFile("weights_start.txt");

            for (int i = 0; i < Params.NOF_EPOCHS; ++i)
            {
                TrainingRun(i);
                if (i % Params.SAVE_WEIGHT_EVERY_XTH_EPOCH == 0)
                    currentNN.SaveWeightsToFile("weights_net_" + ((int)(i / Params.SAVE_WEIGHT_EVERY_XTH_EPOCH)).ToString() + ".txt");
            }
        }
        public void ValidateOuput()
        {
            TicTacToeGame game = new TicTacToeGame();
            game.DoMove(Tuple.Create(3, 2));
            game.DoMove(Tuple.Create(0, 1));
            game.DoMove(Tuple.Create(0, 4)); // y, x
            game.DoMove(Tuple.Create(2, 4));

            Tuple<float[],float> prediction = currentNN.Predict(game.position);

            for (int i = 0; i < 5; ++i)
            {
                Console.WriteLine(prediction.Item1[i * 5 + 0].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 1].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 2].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 3].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 4].ToString("0.000"));
            }
            Console.WriteLine("Value " + prediction.Item2);
            Console.WriteLine("\n");
        }
        public void CheckPerformanceVsRandomKeras(int nofGames)
        {
            bestNN = new NeuralNetwork(currentNN.weights);

            List<NeuralNetwork>  nns = new List<NeuralNetwork>();
            List<NeuralNetwork>  currnns = new List<NeuralNetwork>();

            for (int i = 0; i < nofGames; ++i)
            {
                NeuralNetwork previousNN = new NeuralNetwork(bestNN.weights);
                nns.Add(previousNN);

                NeuralNetwork newNN = new NeuralNetwork(currentNN.weights);
                currnns.Add(newNN);
            }

            Console.WriteLine("Main Thread: CPU run vs random player starting...");
            Params.DIRICHLET_NOISE_WEIGHT = 0.0f;
            List<float> winsVsRand = new List<float>(new float[nofGames]);
            Parallel.For(0, nofGames, new ParallelOptions { MaxDegreeOfParallelism = Params.MAX_THREADS_CPU }, i =>
            {
                Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;
                winsVsRand[i] += PlayAgainstRandom(currnns[i], evaluationNetworkPlayer, Params.NOF_SIMS_PER_MOVE_VS_RANDOM1);
            });
            winrateVsRandTotal1 = (float)winsVsRand.Average();
            winrateVsRandMovingAvg1.ComputeAverage((decimal)winrateVsRandTotal1);
            Console.WriteLine("Main Thread: Wins/Games vs Random Player (" + Params.NOF_SIMS_PER_MOVE_VS_RANDOM1 + " Nodes): " + Math.Round(winrateVsRandTotal1 * 100, 2) + "%");

            winsVsRand = new List<float>(new float[nofGames]);
            Parallel.For(0, nofGames, new ParallelOptions { MaxDegreeOfParallelism = Params.MAX_THREADS_CPU }, i =>
            {
                Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;
                winsVsRand[i] += PlayAgainstRandom(currnns[i], evaluationNetworkPlayer, Params.NOF_SIMS_PER_MOVE_VS_RANDOM2);
            });
            winrateVsRandTotal2 = (float)winsVsRand.Average();
            winrateVsRandMovingAvg2.ComputeAverage((decimal)winrateVsRandTotal2);
            Console.WriteLine("Main Thread: Wins/Games vs Random Player (" + Params.NOF_SIMS_PER_MOVE_VS_RANDOM2 + " Nodes): " + Math.Round(winrateVsRandTotal2 * 100, 2) + "%");

            winsVsRand = new List<float>(new float[nofGames]);
            Parallel.For(0, nofGames, new ParallelOptions { MaxDegreeOfParallelism = Params.MAX_THREADS_CPU }, i =>
            {
                Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;
                winsVsRand[i] += PlayAgainstRandom(currnns[i], evaluationNetworkPlayer, Params.NOF_SIMS_PER_MOVE_VS_RANDOM3);
            });
            winrateVsRandTotal3 = (float)winsVsRand.Average();
            winrateVsRandMovingAvg3.ComputeAverage((decimal)winrateVsRandTotal3);
            Console.WriteLine("Main Thread: Wins/Games vs Random Player (" + Params.NOF_SIMS_PER_MOVE_VS_RANDOM3 + " Nodes): " + Math.Round(winrateVsRandTotal3 * 100, 2) + "%");
        }
        /// <summary>
        /// Return file name for games
        /// </summary>
        /// <param name="nofGames"></param>
        /// <returns></returns>
        public String ProduceTrainingGamesKeras(int nofGames)
        {
            Console.WriteLine("Main Thread: Creating " + nofGames + " training samples...");

            bestNN = new NeuralNetwork(currentNN.weights);

            List<NeuralNetwork> nns = new List<NeuralNetwork>();
            List<NeuralNetwork> currnns = new List<NeuralNetwork>();

            for (int i = 0; i < nofGames; ++i)
            {
                NeuralNetwork playingNNlocal = new NeuralNetwork(currentNN.weights);
                nns.Add(playingNNlocal);

                NeuralNetwork currNNlocal = new NeuralNetwork(currentNN.weights);
                currnns.Add(currNNlocal);
            }

            List<List<Tuple<int,int>>> moves = new List<List<Tuple<int, int>>>(nofGames);
            List<List<List<float>>> policies = new List<List<List<float>>>(nofGames);
            for (int i = 0; i < nofGames; ++i)
            {
                moves.Add(new List<Tuple<int, int>>());
                policies.Add(new List<List<float>>());
            }
            List<float> scores = new List<float>(nofGames);
            scores.AddRange(Enumerable.Repeat(0.0f, nofGames));
            Params.DIRICHLET_NOISE_WEIGHT = 0.5f;
            Parallel.For(0, nofGames, new ParallelOptions { MaxDegreeOfParallelism = Params.MAX_THREADS_CPU }, i =>
            {
                Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;

                scores[i] = RecordOneGame(moves[i], policies[i], evaluationNetworkPlayer, currnns[i], nns[i], true);
            });

            // store moves in file
            Int32 unixTimestamp = (Int32)(DateTime.UtcNow.Subtract(new DateTime(1970, 1, 1))).TotalSeconds;
            String filename = "training_games_" + unixTimestamp + ".txt";
            StreamWriter fileWriter = new StreamWriter("./../../../Training/" + filename);
            for (int i = 0; i < scores.Count; ++i)
            {
                fileWriter.Write(scores[i]+ "\n");
                for (int j = 0; j < moves[i].Count-1; ++j)
                {
                    fileWriter.Write(moves[i][j] + ", ");
                }
                fileWriter.Write(moves[i][moves[i].Count - 1]);
                fileWriter.Write("\n");

                for (int j = 0; j < policies[i].Count; ++j)
                {
                    fileWriter.Write("(");
                    for (int k = 0; k < policies[i][j].Count-1; ++k)
                    {
                        fileWriter.Write(policies[i][j][k] + ", ");
                    }
                    fileWriter.Write(policies[i][j][policies[i][j].Count - 1]);
                    if (j == policies[i].Count - 1)
                    {
                        fileWriter.Write(")");
                    }
                    else
                    {
                        fileWriter.Write("), ");
                    }
                }

                fileWriter.Write("\n");
            }
            fileWriter.Close();
            return filename;
        }
        public void TrainingRun(int run)
        {
            Console.WriteLine("Main Thread: Epoch start");
            //################################# GENERATE NEW WEIGHTS ###########################################

            Console.WriteLine("Main Thread: Creating new offspring weights...");
            Stopwatch sw = new Stopwatch();
            sw.Start();

            List<float> rewards = new List<float>(Params.NOF_OFFSPRING);
            List<List<float>> weights = new List<List<float>>(Params.NOF_OFFSPRING);
            List<List<float>> noise = new List<List<float>>(Params.NOF_OFFSPRING);

            /*create noise*/
            for (int j = 0; j < Params.NOF_OFFSPRING / 2; ++j)
            {
                List<float> temp1 = new List<float>();

                for (int k = 0; k < currentNN.weights.Count; ++k)
                {
                    temp1.Add(RandomNr.GetGaussianFloat());
                }
                noise.Add(temp1);

                /* add the negative of the previous vector (so the perturbations are balanced in all directions)*/
                List<float> temp2 = new List<float>();
                for (int k = 0; k < currentNN.weights.Count; ++k)
                {
                    temp2.Add(-temp1[k]);
                }
                noise.Add(temp2);
            }

            //########################## GENERATE NEURAL NETWORKS FOR TRAINING ##################################

            List<NeuralNetwork> nns = new List<NeuralNetwork>();
            List<NeuralNetwork> currnns = new List<NeuralNetwork>();

            for (int i = 0; i < Params.NOF_OFFSPRING; ++i)
            {
                rewards.Add(0);
                weights.Add(new List<float>());

                /* create new weights for this network */
                weights[i] = new List<float>(currentNN.weights);

                for (int j = 0; j < weights[i].Count; ++j) {
                    weights[i][j] += Params.NOISE_SIGMA * noise[i][j];
                }
                NeuralNetwork playingNNlocal = new NeuralNetwork(weights[i], currentNN.untrainable_weights);
                nns.Add(playingNNlocal);

                NeuralNetwork currNNlocal = new NeuralNetwork(currentNN.weights, currentNN.untrainable_weights);
                currnns.Add(currNNlocal);
            }

            // ################################# COPY WEIGHTS TO GPU MEMORY ###########################################

            if (Params.GPU_ENABLED)
            {
                ID.ResetGlobalID();
                OpenCL.ClearWeights();

                for (int i = 0; i < Params.NOF_OFFSPRING; ++i)
                {
                    nns[i].OpenCLInit(ID.GetGlobalID());
                    currnns[i].OpenCLInit(ID.GetGlobalID());
                }
                OpenCL.CreateNetworkWeightBuffers();
            }

            // ###################################### GPU TRAINING LOOP ##############################################
            sw.Stop();
            Console.WriteLine("Main Thread: Finished in: " + sw.ElapsedMilliseconds +"ms");
            sw.Reset();
            sw.Start();

            Params.DIRICHLET_NOISE_WEIGHT = 0.2f;
            if (Params.GPU_ENABLED)
            {
                Console.WriteLine("Main Thread: GPU training games starting...");
                Thread thread = new Thread(OpenCL.Run);
                thread.Priority = ThreadPriority.Highest;
                thread.Start();

                int numOfThreads = Math.Min(Params.NOF_OFFSPRING, Params.MAX_THREADS_CPU);
                WaitHandle[] waitHandles = new WaitHandle[numOfThreads];

                for (int i = 0; i < numOfThreads; i++)
                {
                    var jk = i;
                    // Or you can use AutoResetEvent/ManualResetEvent
                    var handle = new EventWaitHandle(false, EventResetMode.ManualReset);
                    var thread2 = new Thread(() =>
                    {
                        /* get reward of network*/
                        List<Tuple<int, int>> history = new List<Tuple<int, int>>();
                        float totalReward = 0;
                        for (int j = 0; j < Params.NOF_GAMES_PER_OFFSPRING; ++j) // if more than 2 games we need some noise
                        {
                            history.Clear();
                            Player evaluationNetworkPlayer = (j % 2) == 0 ? Player.X : Player.Z;

                            int result = PlayOneGameGPU(history, evaluationNetworkPlayer, nns[jk], currnns[jk], true);

                            if (evaluationNetworkPlayer == Player.X && result == 1 ||
                                evaluationNetworkPlayer == Player.Z && result == -1)
                            {
                                totalReward++;
                            }
                            else if (result != 0)
                            {
                                totalReward--;
                            }
                            // draw is +0
                        }

                        rewards[jk] = totalReward;

                        handle.Set();
                    });
                    waitHandles[jk] = handle;
                    thread2.Start();
                }
                WaitHandle.WaitAll(waitHandles);
                thread.Abort();
            }

            // ###################################### CPU TRAINING LOOP ##############################################

            else
            {
                Console.WriteLine("Main Thread: CPU training games starting...");
                Parallel.For(0, Params.NOF_OFFSPRING, new ParallelOptions { MaxDegreeOfParallelism = Params.MAX_THREADS_CPU }, i =>
                {
                    NeuralNetwork playingNNlocal = nns[i];
                    NeuralNetwork currNNlocal = currnns[i];

                    /* get reward of network*/
                    List<Tuple<int, int>> history = new List<Tuple<int, int>>();
                    float totalReward = 0;
                    for (int j = 0; j < Params.NOF_GAMES_PER_OFFSPRING; ++j) // if more than 2 games we need some noise
                    {
                        history.Clear();
                        Player evaluationNetworkPlayer = (j % 2) == 0 ? Player.X : Player.Z;

                        int result = PlayOneGame(history, evaluationNetworkPlayer, playingNNlocal, currNNlocal, true);

                        if (evaluationNetworkPlayer == Player.X && result == 1 ||
                            evaluationNetworkPlayer == Player.Z && result == -1)
                        {
                            totalReward++;
                        }
                        else if (result != 0)
                        {
                            totalReward--;
                        }
                        // draw is +0
                    }

                    rewards[i] = totalReward;
                });
            }
            sw.Stop();
            Console.WriteLine("Main Thread: Finished in: " + sw.ElapsedMilliseconds + "ms");
            sw.Reset();
            sw.Start();

            // ########################## CREATE NEW NETWORK GIVEN REWARDS FROM TRAINING LOOP ###########################

            Console.WriteLine("Main Thread: Gathering rewards of training games...");
            // rewards[i] is total reward for the games of player i
            for (int i = 0; i < rewards.Count; ++i)
            {
                //rewards[i] = (rewards[i] > 0) ? rewards[i] : 0; // set reward to 0 if negative
            }
            Console.WriteLine("Main Thread: Average training reward: " + Math.Round(rewards.Average(), 2));

            /* normalize rewards */
            float mean = rewards.Average();
            float stddev = 0.0f;
            for (int i = 0; i < rewards.Count; ++i)
            {
                rewards[i] -= mean;
                stddev += (rewards[i]-mean) * (rewards[i]-mean);
            }
            stddev /= rewards.Count;
            stddev = (float)Math.Sqrt(stddev + Params.EPS);

            for (int i = 0; i < rewards.Count; ++i)
            {
                rewards[i] /= stddev;
            }

            /* set weight for new network */
            for (int j = 0; j < currentNN.weights.Count; ++j)
            {
                float offset = 0.0f;
                for (int k = 0; k < Params.NOF_OFFSPRING; ++k)
                {
                    offset += rewards[k] * noise[k][j];
                }
                currentNN.weights[j] += (Params.LEARNING_RATE / (Params.NOF_OFFSPRING * Params.NOISE_SIGMA)) * offset;
            }
            currentNN.ParseWeights();

            // ######################## RUN TEST GAMES TO CHECK IF NEW NETWORK IS BETTER ###############################

            Console.WriteLine("Main Thread: Combining offspring networks...");
            List<int> wins = new List<int>(new int[Params.NOF_GAMES_TEST]);
            List<int> draws = new List<int>(new int[Params.NOF_GAMES_TEST]);
            List<int> losses = new List<int>(new int[Params.NOF_GAMES_TEST]);
            List<int> movecount = new List<int>(new int[Params.NOF_GAMES_TEST]);
            List<int> winsX = new List<int>(new int[Params.NOF_GAMES_TEST]);
            List<int> winsZ = new List<int>(new int[Params.NOF_GAMES_TEST]);

            nns = new List<NeuralNetwork>();
            currnns = new List<NeuralNetwork>();

            for (int i = 0; i < Params.NOF_GAMES_TEST; ++i)
            {
                NeuralNetwork previousNN = new NeuralNetwork(bestNN.weights, bestNN.untrainable_weights);
                nns.Add(previousNN);

                NeuralNetwork newNN = new NeuralNetwork(currentNN.weights, currentNN.untrainable_weights);
                currnns.Add(newNN);
            }

            // ################################# COPY WEIGHTS TO GPU MEMORY ###########################################

            if (Params.GPU_ENABLED)
            {
                ID.ResetGlobalID();
                OpenCL.ClearWeights();

                for (int i = 0; i < Params.NOF_GAMES_TEST; ++i)
                {
                    nns[i].OpenCLInit(ID.GetGlobalID());
                    currnns[i].OpenCLInit(ID.GetGlobalID());
                }
                OpenCL.CreateNetworkWeightBuffers();
            }
            sw.Stop();
            Console.WriteLine("Main Thread: Finished in: " + sw.ElapsedMilliseconds + "ms");
            sw.Reset();
            sw.Start();

            // #################################### GPU TEST LOOP ##########################################

            if (Params.GPU_ENABLED)
            {
                Console.WriteLine("Main Thread: GPU test games starting...");

                Params.DIRICHLET_NOISE_WEIGHT = 0.2f;
                Thread thread = new Thread(OpenCL.Run);
                thread.Priority = ThreadPriority.Highest;
                thread.Start();

                int numOfThreads = Math.Min(Params.NOF_GAMES_TEST, Params.MAX_THREADS_CPU);
                WaitHandle[] waitHandles = new WaitHandle[numOfThreads];

                for (int i = 0; i < numOfThreads; i++)
                {
                    var jk = i;
                    // Or you can use AutoResetEvent/ManualResetEvent
                    var handle = new EventWaitHandle(false, EventResetMode.ManualReset);
                    var thread2 = new Thread(() =>
                    {
                        List<Tuple<int, int>> history = new List<Tuple<int, int>>();
                        Player evaluationNetworkPlayer = (jk % 2) == 0 ? Player.X : Player.Z;

                        int result = PlayOneGameGPU(history, evaluationNetworkPlayer, currnns[jk], nns[jk], false);

                        if (result == 1)
                            winsX[jk]++;
                        else if (result == -1)
                            winsZ[jk]++;

                        if (evaluationNetworkPlayer == Player.X && result == 1 ||
                            evaluationNetworkPlayer == Player.Z && result == -1)
                            wins[jk]++;
                        else if (result == 0)
                            draws[jk]++;
                        else
                            losses[jk]++;

                        movecount[jk] += history.Count;

                        /* to display some games (debugging)*/
                        if (run % Params.SHOW_SAMPLE_MATCHES_EVERY_XTH_EPOCH == 0 && jk >= Params.NOF_GAMES_TEST - 2)
                        {
                            if(jk % 2 == 0)
                            {
                                Console.WriteLine("Sample game where Evaluation Player is X");
                            }
                            else
                            {
                                Console.WriteLine("Sample game where Evaluation Player is Z");
                            }
                            TicTacToeGame game = new TicTacToeGame();
                            game.DisplayHistory(history);
                        }
                        handle.Set();
                    });
                    waitHandles[jk] = handle;
                    thread2.Start();
                }
                WaitHandle.WaitAll(waitHandles);
                thread.Abort();
            }

            // #################################### CPU TEST LOOP ##########################################
            else
            {
                Console.WriteLine("Main Thread: CPU test games starting...");
                Params.DIRICHLET_NOISE_WEIGHT = 0.2f;
                Parallel.For(0, Params.NOF_GAMES_TEST, new ParallelOptions { MaxDegreeOfParallelism = Params.MAX_THREADS_CPU }, i =>
                {
                    List<Tuple<int, int>> history = new List<Tuple<int, int>>();
                    Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;

                    int result = PlayOneGame(history, evaluationNetworkPlayer, currnns[i], nns[i], false);

                    if (result == 1)
                        winsX[i]++;
                    else if (result == -1)
                        winsZ[i]++;

                    if (evaluationNetworkPlayer == Player.X && result == 1 ||
                        evaluationNetworkPlayer == Player.Z && result == -1)
                        wins[i]++;
                    else if (result == 0)
                        draws[i]++;
                    else
                        losses[i]++;

                    movecount[i] += history.Count;

                    /* to display some games (debugging)*/
                    if (run % Params.SHOW_SAMPLE_MATCHES_EVERY_XTH_EPOCH == 0 && i >= Params.NOF_GAMES_TEST - 2)
                    {
                        if((i % 2) == 0)
                            Console.WriteLine("Eval player playing as Player X");
                        else
                            Console.WriteLine("Eval player playing as Player Z");
                        TicTacToeGame game = new TicTacToeGame();
                        game.DisplayHistory(history);
                    }
                });
            }
            sw.Stop();
            Console.WriteLine("Main Thread: Finished in: " + sw.ElapsedMilliseconds + "ms");
            sw.Reset();
            sw.Start();

            // #################################### PROCESS STATISTICS ##########################################

            Console.WriteLine("Main Thread: Process test run statistics, creating new best network...");
            int winsTotal = wins.Sum();
            int lossesTotal = losses.Sum();
            int drawsTotal = draws.Sum();
            int winsAsXtotal = winsX.Sum();
            int winsAsZtotal = winsZ.Sum();
            int totalMoves = movecount.Sum();

            decimal nofgames = Params.NOF_GAMES_TEST;
            winsAsXMovingAvg.ComputeAverage(winsAsXtotal / (decimal)nofgames);
            winsAsZMovingAvg.ComputeAverage(winsAsZtotal / (decimal)nofgames);
            drawsMovingAvg.ComputeAverage(drawsTotal / (decimal)nofgames);
            averageMovesMovingAvg.ComputeAverage(totalMoves / (decimal)nofgames);

            Console.WriteLine("Main Thread: Winrate vs best network " + Math.Round(((float)(winsTotal) / Params.NOF_GAMES_TEST) * 100,2) + "%");
            Console.WriteLine("Score: W/D/L " + winsTotal + "/" + drawsTotal + "/" + lossesTotal + "  WinrateX/Drawrate/WinrateZ " +
                Math.Round(winsAsXMovingAvg.Average, 2) + "/" + Math.Round(drawsMovingAvg.Average, 2) + "/" + Math.Round(winsAsZMovingAvg.Average, 2));


            // #################################### CREATE NEW BEST NETWORK ##########################################
            if (((winsTotal+drawsTotal*0.5)/ Params.NOF_GAMES_TEST)*100.0f < Params.MINIMUM_WIN_PERCENTAGE)
            {
                // bad new network, ignore it
                currentPseudoELO += 0;

                bestNN.ApplyWeightDecay(); // every time the new network is not better take old one with decayed weights
                bestNN.ParseWeights();
                currentNN.weights = new List<float>(bestNN.weights);
                currentNN.untrainable_weights = new List<float>(bestNN.untrainable_weights);

                currentNN.ParseWeights();
            }
            else
            {
                // new best network, use it
                currentPseudoELO += (float)(winsTotal - lossesTotal) / (float)Params.NOF_GAMES_TEST;

                bestNN.weights = new List<float>(currentNN.weights);
                bestNN.untrainable_weights = new List<float>(currentNN.untrainable_weights);
                bestNN.ParseWeights();

                printPolicy(bestNN);
                //printValue(bestNN);
            }
            sw.Stop();
            Console.WriteLine("Main Thread: Finished in: " + sw.ElapsedMilliseconds + "ms");
            sw.Reset();
            sw.Start();

            // #################################### CHECK PERFORMANCE VS RANDOM ##########################################

            if (winsTotal >= lossesTotal || winrateVsRandTotal1 < 0.0f)
            {
                Console.WriteLine("Main Thread: CPU run vs random player starting...");
                Params.DIRICHLET_NOISE_WEIGHT = 0.0f;
                List<float> winsVsRand = new List<float>(new float[Params.NOF_GAMES_VS_RANDOM]);
                Parallel.For(0, Params.NOF_GAMES_VS_RANDOM, new ParallelOptions { MaxDegreeOfParallelism = Params.MAX_THREADS_CPU }, i =>
                {
                    Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;
                    winsVsRand[i] += PlayAgainstRandom(currnns[i], evaluationNetworkPlayer, Params.NOF_SIMS_PER_MOVE_VS_RANDOM1);
                });
                winrateVsRandTotal1 = (float)winsVsRand.Average();
                winrateVsRandMovingAvg1.ComputeAverage((decimal)winrateVsRandTotal1);
                Console.WriteLine("Main Thread: Wins/Games vs Random Player (" + Params.NOF_SIMS_PER_MOVE_VS_RANDOM1 + " Nodes): " + Math.Round(winrateVsRandTotal1 * 100,2) + "%");

                winsVsRand = new List<float>(new float[Params.NOF_GAMES_VS_RANDOM]);
                Parallel.For(0, Params.NOF_GAMES_VS_RANDOM, new ParallelOptions { MaxDegreeOfParallelism = Params.MAX_THREADS_CPU }, i =>
                {
                    Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;
                    winsVsRand[i] += PlayAgainstRandom(currnns[i], evaluationNetworkPlayer, Params.NOF_SIMS_PER_MOVE_VS_RANDOM2);
                });
                winrateVsRandTotal2 = (float)winsVsRand.Average();
                winrateVsRandMovingAvg2.ComputeAverage((decimal)winrateVsRandTotal2);
                Console.WriteLine("Main Thread: Wins/Games vs Random Player (" + Params.NOF_SIMS_PER_MOVE_VS_RANDOM2 + " Nodes): " + Math.Round(winrateVsRandTotal2 * 100,2) + "%");

                winsVsRand = new List<float>(new float[Params.NOF_GAMES_VS_RANDOM]);
                Parallel.For(0, Params.NOF_GAMES_VS_RANDOM, new ParallelOptions { MaxDegreeOfParallelism = Params.MAX_THREADS_CPU }, i =>
                {
                    Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;
                    winsVsRand[i] += PlayAgainstRandom(currnns[i], evaluationNetworkPlayer, Params.NOF_SIMS_PER_MOVE_VS_RANDOM3);
                });
                winrateVsRandTotal3 = (float)winsVsRand.Average();
                winrateVsRandMovingAvg3.ComputeAverage((decimal)winrateVsRandTotal3);
                Console.WriteLine("Main Thread: Wins/Games vs Random Player (" + Params.NOF_SIMS_PER_MOVE_VS_RANDOM3 + " Nodes): " + Math.Round(winrateVsRandTotal3 * 100,2) + "%");
                sw.Stop();
                Console.WriteLine("Main Thread: Finished in: " + sw.ElapsedMilliseconds + "ms");
            }

            // #################################### WRITE PLOTTING STATISTICS ##########################################

            using (System.IO.StreamWriter file = new System.IO.StreamWriter(Params.PLOT_FILENAME, true))
            {
                file.WriteLine(currentPseudoELO + " " + Math.Round(winsAsXMovingAvg.Average, 2) + " " +
                    Math.Round(winsAsZMovingAvg.Average, 2) + " " + Math.Round(drawsMovingAvg.Average, 2) + " " +
                    Math.Round(averageMovesMovingAvg.Average, 2) + " " + Math.Round(winrateVsRandMovingAvg1.Average, 2)
                    + " " + Math.Round(winrateVsRandMovingAvg2.Average, 2) + " " + Math.Round(winrateVsRandMovingAvg3.Average, 2));
            }
        }
        /// <summary>
        /// Play against random player
        /// </summary>
        /// <param name="nofGames"></param>
        /// <returns>1 if game is won by NN player</returns>
        public float PlayAgainstRandom(NeuralNetwork NN, Player evaluationNetworkPlayer, int nofSimsPerMove)
        {
            List<Tuple<int, int>> history = new List<Tuple<int, int>>();

            TicTacToeGame game = new TicTacToeGame();

            float result = 0.0f;
            Node<TicTacToePosition> MCTSRootNode = new Node<TicTacToePosition>(null);

            for (int curr_ply = 0; curr_ply < Params.MAXIMUM_PLYS; ++curr_ply)  // we always finish the game for tic tac toe
            {
                MCTSRootNode.Value = new TicTacToePosition(game.position);

                if (game.IsOver())
                {
                    result = game.GetScore();
                    //TicTacToeGame game4 = new TicTacToeGame();
                    //game4.DisplayHistory(history);
                    if (evaluationNetworkPlayer == Player.X && result == 1 ||
                        evaluationNetworkPlayer == Player.Z && result == -1)
                    {
                        return 1;
                    }
                    else
                        return 0;
                }

                DirichletNoise dn = new DirichletNoise(game.GetMoves().Count);
                // for root node (all root nodes not just the actual game start)
                // also tree use makes this a bit less effective going down the tree, maybe use temperature later

                int best_child_index = -1;
                if (game.position.sideToMove == evaluationNetworkPlayer)
                {
                    for (int simulation = 0; simulation < nofSimsPerMove; ++simulation)
                    {
                        SearchUsingNN(MCTSRootNode, NN);
                    }
                    //best_child_index = findBestChildWinrate(MCTSRootNode, dn, curr_ply);
                    best_child_index = findBestChildVisitCount(MCTSRootNode, dn, curr_ply);
                }
                else
                {
                    SearchUsingNN(MCTSRootNode, NN); // just in case we dont create the children properly for random player
                    best_child_index = RandomGen2.Next(0, MCTSRootNode.Children.Count);
                }

                List<Tuple<int, int>> moves = game.GetMoves();
                //DisplayMCTSTree(MCTSRootNode);
                Tuple<int, int> move = moves[best_child_index];
                game.DoMove(move);
                history.Add(move);

                /* tree re-use */
                MCTSRootNode = MCTSRootNode.Children[best_child_index];
                MCTSRootNode.parent = null; // remove the tree above new root -> free memory, don't propagate wins 
            }

            result = game.position.score;

            if (evaluationNetworkPlayer == Player.X && result == 1 ||
                evaluationNetworkPlayer == Player.Z && result == -1)
            {
                return 1;
            }
            return 0;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="history"></param>
        /// <param name="aEvaluationNetworkPlayer"></param>
        /// <param name="NN1"></param>
        /// <param name="NN2"></param>
        /// <returns>Return 0 for draw, win for X 1, win for Z -1 </returns>
        public int PlayOneGame(List<Tuple<int, int>> history, Player aEvaluationNetworkPlayer, NeuralNetwork NN1, NeuralNetwork NN2, bool train)
        {
            TicTacToeGame game = new TicTacToeGame();
            Node<TicTacToePosition> MCTSRootNodeNN1 = new Node<TicTacToePosition>(null);
            Node<TicTacToePosition> MCTSRootNodeNN2 = new Node<TicTacToePosition>(null);

            for (int curr_ply = 0; curr_ply < Params.MAXIMUM_PLYS; ++curr_ply)  // we always finish the game for tic tac toe
            {
                MCTSRootNodeNN1.Value = new TicTacToePosition(game.position);
                MCTSRootNodeNN2.Value = new TicTacToePosition(game.position);

                if (game.IsOver()) {
                    return game.GetScore();
                }

                DirichletNoise dn = new DirichletNoise(game.GetMoves().Count); // for root node (all root nodes not just the actual game start)
                                                                               // also tree use makes this a bit less effective going down the tree, maybe use temperature later

                createChildren(MCTSRootNodeNN1);
                /* find value, policy */
                if (MCTSRootNodeNN1.nn_policy == null)
                {
                    calculateNNOutput(MCTSRootNodeNN1, NN1);
                }
                createChildren(MCTSRootNodeNN2);
                if (MCTSRootNodeNN2.nn_policy == null)
                {
                    calculateNNOutput(MCTSRootNodeNN2, NN2);
                }
                int nofSimsPerMove = train ? Params.NOF_SIMS_PER_MOVE_TRAINING : Params.NOF_SIMS_PER_MOVE_TESTING;

                for (int simulation = 0; simulation < nofSimsPerMove; ++simulation)
                {
                    if(curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                        SearchUsingNN(MCTSRootNodeNN1, NN1); // expand tree and improve accuracy at MCTSRootNode
                    else
                        SearchUsingNN(MCTSRootNodeNN2, NN2); // expand tree and improve accuracy at MCTSRootNode

                    // show last simulation tree
                    if (simulation == nofSimsPerMove - 1 && curr_ply == 0)
                    {
                        //DisplayMCTSTree(MCTSRootNode);
                    }
                }

                int best_child_index = -1;
                if (curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                    //best_child_index = findBestChildWinrate(MCTSRootNodeNN1, dn, curr_ply);
                    best_child_index = findBestChildVisitCount(MCTSRootNodeNN1, dn, curr_ply);
                else
                    //best_child_index = findBestChildWinrate(MCTSRootNodeNN2, dn, curr_ply);
                    best_child_index = findBestChildVisitCount(MCTSRootNodeNN2, dn, curr_ply);

                List<Tuple<int, int>> moves = game.GetMoves();
                Tuple<int, int> move = moves[best_child_index]; // add randomness here
                game.DoMove(move);
                history.Add(move);

                /* tree re-use */
                MCTSRootNodeNN1 = MCTSRootNodeNN1.Children[best_child_index];
                MCTSRootNodeNN2 = MCTSRootNodeNN2.Children[best_child_index];
                MCTSRootNodeNN1.parent = null;
                MCTSRootNodeNN2.parent = null;
            }

            return game.position.score;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="history"></param>
        /// <param name="aEvaluationNetworkPlayer"></param>
        /// <param name="NN1"></param>
        /// <param name="NN2"></param>
        /// <returns>Return 0 for draw, win for X 1, win for Z -1 </returns>
        public int RecordOneGame(List<Tuple<int, int>> history, List<List<float>> policies, Player aEvaluationNetworkPlayer, 
            NeuralNetwork NN1, NeuralNetwork NN2, bool train)
        {
            TicTacToeGame game = new TicTacToeGame();
            Node<TicTacToePosition> MCTSRootNodeNN1 = new Node<TicTacToePosition>(null);
            Node<TicTacToePosition> MCTSRootNodeNN2 = new Node<TicTacToePosition>(null);

            for (int curr_ply = 0; curr_ply < Params.MAXIMUM_PLYS; ++curr_ply)  // we always finish the game for tic tac toe
            {
                MCTSRootNodeNN1.Value = new TicTacToePosition(game.position);
                MCTSRootNodeNN2.Value = new TicTacToePosition(game.position);

                if (game.IsOver())
                {
                    return game.GetScore();
                }

                DirichletNoise dn = new DirichletNoise(game.GetMoves().Count); // for root node (all root nodes not just the actual game start)
                                                                               // also tree use makes this a bit less effective going down the tree, maybe use temperature later

                createChildren(MCTSRootNodeNN1);
                /* find value, policy */
                if (MCTSRootNodeNN1.nn_policy == null)
                {
                    calculateNNOutput(MCTSRootNodeNN1, NN1);
                }
                createChildren(MCTSRootNodeNN2);
                if (MCTSRootNodeNN2.nn_policy == null)
                {
                    calculateNNOutput(MCTSRootNodeNN2, NN2);
                }
                int nofSimsPerMove = train ? Params.NOF_SIMS_PER_MOVE_TRAINING : Params.NOF_SIMS_PER_MOVE_TESTING;

                for (int simulation = 0; simulation < nofSimsPerMove; ++simulation)
                {
                    if (curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                        SearchUsingNN(MCTSRootNodeNN1, NN1); // expand tree and improve accuracy at MCTSRootNode
                    else
                        SearchUsingNN(MCTSRootNodeNN2, NN2); // expand tree and improve accuracy at MCTSRootNode

                    // show last simulation tree
                    if (simulation == nofSimsPerMove - 1 && curr_ply == 0)
                    {
                        //DisplayMCTSTree(MCTSRootNode);
                    }
                }

                // after search, record root node visits for new policy vector
                if (curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                {
                    List<float> policy = Enumerable.Repeat(0.0f, 25).ToList();
                    float totalVisits = 0;
                    for (int i = 0; i < MCTSRootNodeNN1.Children.Count; ++i)
                    {
                        policy[MCTSRootNodeNN1.Children[i].moveIndex] = MCTSRootNodeNN1.Children[i].visitCount;
                        totalVisits += MCTSRootNodeNN1.Children[i].visitCount;
                    }
                    for (int i = 0; i < MCTSRootNodeNN1.Children.Count; ++i)
                    {
                        policy[MCTSRootNodeNN1.Children[i].moveIndex] /= totalVisits;
                        if(policy[MCTSRootNodeNN1.Children[i].moveIndex] == float.NaN ||
                            policy[MCTSRootNodeNN1.Children[i].moveIndex] == float.NegativeInfinity ||
                            policy[MCTSRootNodeNN1.Children[i].moveIndex] == float.PositiveInfinity)
                        {
                            policy[MCTSRootNodeNN1.Children[i].moveIndex] = 0.0f;
                        }
                    }
                    policies.Add(policy);
                }
                else
                { 
                    List<float> policy = new List<float>(new float[25]);
                    float totalVisits = 0;
                    for (int i = 0; i < MCTSRootNodeNN2.Children.Count; ++i)
                    {
                        policy[MCTSRootNodeNN2.Children[i].moveIndex] = MCTSRootNodeNN2.Children[i].visitCount;
                        totalVisits += MCTSRootNodeNN2.Children[i].visitCount;
                    }
                    for (int i = 0; i < MCTSRootNodeNN2.Children.Count; ++i)
                    {
                        policy[MCTSRootNodeNN2.Children[i].moveIndex] /= totalVisits;
                        if (policy[MCTSRootNodeNN1.Children[i].moveIndex] == float.NaN ||
                            policy[MCTSRootNodeNN1.Children[i].moveIndex] == float.NegativeInfinity ||
                            policy[MCTSRootNodeNN1.Children[i].moveIndex] == float.PositiveInfinity)
                        {
                            policy[MCTSRootNodeNN1.Children[i].moveIndex] = 0.0f;
                        }
                    }

                    policies.Add(policy);
                }

                int best_child_index = -1;
                if (curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                    //best_child_index = findBestChildWinrate(MCTSRootNodeNN1, dn, curr_ply);
                    best_child_index = findBestChildVisitCount(MCTSRootNodeNN1, dn, curr_ply);
                else
                    //best_child_index = findBestChildWinrate(MCTSRootNodeNN2, dn, curr_ply);
                    best_child_index = findBestChildVisitCount(MCTSRootNodeNN2, dn, curr_ply);

                List<Tuple<int, int>> moves = game.GetMoves();
                Tuple<int, int> move = moves[best_child_index]; // add randomness here
                game.DoMove(move);
                history.Add(move);

                /* tree re-use */
                MCTSRootNodeNN1 = MCTSRootNodeNN1.Children[best_child_index];
                MCTSRootNodeNN2 = MCTSRootNodeNN2.Children[best_child_index];
                MCTSRootNodeNN1.parent = null;
                MCTSRootNodeNN2.parent = null;
            }

            return game.position.score;
        }
        public int PlayOneGameGPU(List<Tuple<int, int>> history, Player aEvaluationNetworkPlayer, NeuralNetwork NN1, NeuralNetwork NN2, bool train)
        {
            TicTacToeGame game = new TicTacToeGame();
            Node<TicTacToePosition> MCTSRootNodeNN1 = new Node<TicTacToePosition>(null);
            Node<TicTacToePosition> MCTSRootNodeNN2 = new Node<TicTacToePosition>(null);

            Queue<Node<TicTacToePosition>> pendingNN1Requests = new Queue<Node<TicTacToePosition>>();
            Queue<Node<TicTacToePosition>> pendingNN2Requests = new Queue<Node<TicTacToePosition>>();

            for (int curr_ply = 0; curr_ply < Params.MAXIMUM_PLYS; ++curr_ply)  // we always finish the game for tic tac toe
            {
                MCTSRootNodeNN1.Value = new TicTacToePosition(game.position);
                MCTSRootNodeNN2.Value = new TicTacToePosition(game.position);

                if (game.IsOver())
                {
                    return game.GetScore();
                }
                createChildren(MCTSRootNodeNN1);
                createChildren(MCTSRootNodeNN2);
                DirichletNoise dn = new DirichletNoise(game.GetMoves().Count); // for root node (all root nodes not just the actual game start)
                                                                               // also tree use makes this a bit less effective going down the tree, maybe use temperature later

                int maxSimulations = train ? Params.NOF_SIMS_PER_MOVE_TRAINING : Params.NOF_SIMS_PER_MOVE_TESTING;
                
                for (int simulation = 0; simulation < maxSimulations; ++simulation)
                {
                    if (curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                    {
                        Tuple<float[], float> result = NN1.GetResultAsync(); // try to get a result, if there is one, try to get more
                        while (result != null)
                        {
                            Node<TicTacToePosition> nodeToUpdate = pendingNN1Requests.Dequeue();
                            nodeToUpdate.nn_policy = new List<float>(result.Item1);
                            nodeToUpdate.nn_value = result.Item2;
                            nodeToUpdate.waitingForGPUPrediction = false;
                            removeVirtualLoss(nodeToUpdate);
                            backpropagateScore(nodeToUpdate, nodeToUpdate.nn_value);

                            result = NN1.GetResultAsync(); // try to get a result, if there is one, try to get more
                        }
                        while (pendingNN1Requests.Count > Params.MAX_PENDING_NN_EVALS)
                        {
                            // if we need to wait then wait
                            result = NN1.GetResultSync();
                            Node<TicTacToePosition> nodeToUpdate = pendingNN1Requests.Dequeue();
                            nodeToUpdate.nn_policy = new List<float>(result.Item1);
                            nodeToUpdate.nn_value = result.Item2;
                            nodeToUpdate.waitingForGPUPrediction = false;
                            removeVirtualLoss(nodeToUpdate);
                            backpropagateScore(nodeToUpdate, nodeToUpdate.nn_value);
                        }
                        SearchUsingNNGPU(MCTSRootNodeNN1, NN1, pendingNN1Requests); // expand tree and improve accuracy at MCTSRootNode
                    }
                    else
                    {
                        Tuple<float[], float> result = NN2.GetResultAsync(); // try to get a result, if there is one, try to get more
                        while (result != null)
                        {
                            Node<TicTacToePosition> nodeToUpdate = pendingNN2Requests.Dequeue();
                            nodeToUpdate.nn_policy = new List<float>(result.Item1);
                            nodeToUpdate.nn_value = result.Item2;
                            nodeToUpdate.waitingForGPUPrediction = false;
                            removeVirtualLoss(nodeToUpdate);
                            backpropagateScore(nodeToUpdate, nodeToUpdate.nn_value);

                            result = NN2.GetResultAsync(); // try to get a result, if there is one, try to get more
                        }
                        while (pendingNN2Requests.Count > Params.MAX_PENDING_NN_EVALS)
                        {
                            // if we need to wait then wait
                            result = NN2.GetResultSync();
                            Node<TicTacToePosition> nodeToUpdate = pendingNN2Requests.Dequeue();
                            nodeToUpdate.nn_policy = new List<float>(result.Item1);
                            nodeToUpdate.nn_value = result.Item2;
                            nodeToUpdate.waitingForGPUPrediction = false;
                            removeVirtualLoss(nodeToUpdate);
                            backpropagateScore(nodeToUpdate, nodeToUpdate.nn_value);
                        }
                        SearchUsingNNGPU(MCTSRootNodeNN2, NN2, pendingNN2Requests); // expand tree and improve accuracy at MCTSRootNode
                    }

                    // show last simulation tree
                    //if (simulation == Params.NOF_SIMS_PER_MOVE_TESTING - 1 && curr_ply == 0)
                    //{
                    //     DisplayMCTSTree(MCTSRootNode);
                    //}
                }
                // wait for all search results before deciding on which move to play
                while (pendingNN1Requests.Count > 0)
                {
                    // if we need to wait then wait
                    Tuple<float[], float> result = NN1.GetResultSync();
                    Node<TicTacToePosition> nodeToUpdate = pendingNN1Requests.Dequeue();
                    nodeToUpdate.nn_policy = new List<float>(result.Item1);
                    nodeToUpdate.nn_value = result.Item2;
                    nodeToUpdate.waitingForGPUPrediction = false;
                    removeVirtualLoss(nodeToUpdate);
                    backpropagateScore(nodeToUpdate, nodeToUpdate.nn_value);
                }
                while (pendingNN2Requests.Count > 0)
                {
                    // if we need to wait then wait
                    Tuple<float[], float> result = NN2.GetResultSync();
                    Node<TicTacToePosition> nodeToUpdate = pendingNN2Requests.Dequeue();
                    nodeToUpdate.nn_policy = new List<float>(result.Item1);
                    nodeToUpdate.nn_value = result.Item2;
                    nodeToUpdate.waitingForGPUPrediction = false;
                    removeVirtualLoss(nodeToUpdate);
                    backpropagateScore(nodeToUpdate, nodeToUpdate.nn_value);
                }

                int best_child_index = -1;
                if (curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                   // best_child_index = findBestChildWinrate(MCTSRootNodeNN1, dn, curr_ply);
                    best_child_index = findBestChildVisitCount(MCTSRootNodeNN1, dn, curr_ply);
                else
                    //best_child_index = findBestChildWinrate(MCTSRootNodeNN2, dn, curr_ply);
                    best_child_index = findBestChildVisitCount(MCTSRootNodeNN2, dn, curr_ply);

                //int best_child_index = findBestChildVisitCount(MCTSRootNode, dn);

                List<Tuple<int, int>> moves = game.GetMoves();
                Tuple<int, int> move = moves[best_child_index]; // add randomness here
                game.DoMove(move);
                history.Add(move);

                /* tree re-use */
                MCTSRootNodeNN1 = MCTSRootNodeNN1.Children[best_child_index];
                MCTSRootNodeNN2 = MCTSRootNodeNN2.Children[best_child_index];
                MCTSRootNodeNN1.parent = null;
                MCTSRootNodeNN2.parent = null;
            }

            return game.position.score;
        }
        /// <summary>
        /// The Search uses a tree Nodes<TicTacToePosition> and expands it until it runs out of time
        /// Nodes which look promising according to the NN are expanded greedily
        /// </summary>
        /// <param name="currNode"></param>
        /// <returns>Eval</returns>
        private void SearchUsingNN(Node<TicTacToePosition> currNode, NeuralNetwork NN)
        {
            TicTacToeGame game = new TicTacToeGame(currNode.Value);
            List<Tuple<int, int>> moves = game.GetMoves();

            /* find the most promising leaf node */
            currNode = findMostPromisingLeafNode(currNode);

            /* if the leaf node is a game ending state use correct score */
            float score = 0.0f;
            game = new TicTacToeGame(currNode.Value);
            if (game.IsOver())
            {
                if(RandomGen2.Next(0,100) <= Params.PERCENT_GROUND_TRUTH)
                {
                    score = (game.GetScore() + 1.0f) / 2.0f;
                }
                else
                {
                    if (currNode.nn_policy == null)
                    {
                        calculateNNOutput(currNode, NN);
                    }
                    score = currNode.nn_value;
                }
            }
            else
            {
                /* find value, policy */
                if (currNode.nn_policy == null)
                {
                    calculateNNOutput(currNode, NN);
                }

                /* create children of normal leaf */
                createChildren(currNode);

                /* find best child.. */
                currNode = findMostPromisingLeafNode(currNode);

                game = new TicTacToeGame(currNode.Value);
                if (game.IsOver())
                {
                    if (RandomGen2.Next(0, 100) <= Params.PERCENT_GROUND_TRUTH)
                    {
                        score = (game.GetScore() + 1.0f) / 2.0f;
                    }
                    else
                    {
                        if (currNode.nn_policy == null)
                        {
                            calculateNNOutput(currNode, NN);
                        }
                        score = currNode.nn_value;
                    }
                }
                else
                {
                    if (currNode.nn_policy == null)
                    {
                        calculateNNOutput(currNode, NN);
                    }

                    score = currNode.nn_value; // [0..1] where player X wins at 1 and player Z wins at 0
                }
            }

            /* update the tree with the new score and visit counts */
            backpropagateScore(currNode, score);
        }

        /// <summary>
        /// The Search uses a tree Nodes<TicTacToePosition> and expands it until it runs out of time
        /// Nodes which look promising according to the NN are expanded greedily
        /// </summary>
        /// <param name="currNode"></param>
        /// <returns>Eval</returns>
        private void SearchUsingNNGPU(Node<TicTacToePosition> currNode, NeuralNetwork NN, Queue<Node<TicTacToePosition>> queue)
        {
            TicTacToeGame game = new TicTacToeGame(currNode.Value);
            List<Tuple<int, int>> moves = game.GetMoves();

            /* find the most promising leaf node */
            currNode = findMostPromisingLeafNode(currNode);

            /* if the leaf node is a game ending state use correct score */
            game = new TicTacToeGame(currNode.Value);
            if (game.IsOver())
            {
                /* update the tree with the new score and visit counts */
                if (RandomGen2.Next(0, 100) <= Params.PERCENT_GROUND_TRUTH)
                {
                    backpropagateScore(currNode, (game.GetScore() + 1.0f) / 2.0f);
                }
                else
                {
                    if (currNode.nn_policy == null)
                    {
                        calculateNNOutputGPU(currNode, NN, queue);
                    }
                }
            }
            else
            {
                /* find value, policy of the leaf node (for root node only actually, rest is covered below) */
                if (currNode.nn_policy == null && !currNode.waitingForGPUPrediction)
                {
                    calculateNNOutputGPU(currNode, NN, queue);
                }

                /* create children of the leaf (must exist since not game ending state)*/
                createChildren(currNode);

                /*visit the child and propagate value up the tree*/
                currNode = findMostPromisingLeafNode(currNode);

                game = new TicTacToeGame(currNode.Value);
                if (game.IsOver())
                {
                    // [-1, 1] where player X wins at 1 and player Z at -1
                    if (RandomGen2.Next(0, 100) <= Params.PERCENT_GROUND_TRUTH)
                    {
                        backpropagateScore(currNode, (game.GetScore() + 1.0f) / 2.0f);
                    }
                    else
                    {
                        if (currNode.nn_policy == null)
                        {
                            calculateNNOutputGPU(currNode, NN, queue);
                        }
                    }
                }
                else
                {
                    if (currNode.nn_policy == null && !currNode.waitingForGPUPrediction)
                    {
                        calculateNNOutputGPU(currNode, NN, queue);
                    }
                    else
                    {
                        // this isn't possible, we dont visit a non game ending state twice
                        // there are children now, so this cant be a leaf
                    }
                }
            }
        }
        private void propagateVirtualLoss(Node<TicTacToePosition> currNode)
        {
            // we store the winrate for the opposite player in the node, during search we look at the next level
            while (currNode != null)
            {
                currNode.virtualLossCount += 1;
                currNode = currNode.GetParent();
            }
        }
        private void removeVirtualLoss(Node<TicTacToePosition> currNode)
        {
            // we store the winrate for the opposite player in the node, during search we look at the next level
            while (currNode != null)
            {
                currNode.virtualLossCount -= 1;
                currNode = currNode.GetParent();
            }
        }
        private void printPolicy(NeuralNetwork nn)
        {
            Console.WriteLine("\nPolicy of boards");
            TicTacToeGame game = new TicTacToeGame();
            Console.WriteLine(game.position.ToString());
            Tuple<float[], float> prediction = nn.Predict(game.position);

            for (int i = 0; i < 5; ++i)
            { 
                Console.WriteLine(prediction.Item1[i * 5 + 0].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 1].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 2].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 3].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 4].ToString("0.000"));
            }
            Console.WriteLine("Value " + prediction.Item2);
            Console.WriteLine("\n");

            game.DoMove(Tuple.Create(3, 3));
            game.DoMove(Tuple.Create(0, 0));
            game.DoMove(Tuple.Create(3, 2));
            game.DoMove(Tuple.Create(4, 4));
            game.DoMove(Tuple.Create(3, 1));

            Console.WriteLine(game.position.ToString());
            prediction = nn.Predict(game.position);
            for (int i = 0; i < 5; ++i)
            {
                Console.WriteLine(prediction.Item1[i * 5 + 0].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 1].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 2].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 3].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 4].ToString("0.000"));
            }
            Console.WriteLine("Value " + prediction.Item2);
            Console.WriteLine("\n");

            game.DoMove(Tuple.Create(0, 3));

            Console.WriteLine(game.position.ToString());
            prediction = nn.Predict(game.position);
            for (int i = 0; i < 5; ++i)
            {
                Console.WriteLine(prediction.Item1[i * 5 + 0].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 1].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 2].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 3].ToString("0.000") + " " +
                prediction.Item1[i * 5 + 4].ToString("0.000"));
            }
            Console.WriteLine("Value " + prediction.Item2);
            Console.WriteLine("\n");
        }
        private void printValue(NeuralNetwork nn)
        {
            Console.WriteLine("Value of the board after move is played");
            TicTacToeGame game = new TicTacToeGame();

            Node<TicTacToePosition> MCTSRootNode = new Node<TicTacToePosition>(null);
            MCTSRootNode.Value = new TicTacToePosition(game.position);
            Tuple<float[], float> prediction = nn.Predict(MCTSRootNode.Value);
            MCTSRootNode.nn_policy = new List<float>(prediction.Item1);
            MCTSRootNode.nn_value = prediction.Item2;

            createChildren(MCTSRootNode);
            for (int i = 0; i < MCTSRootNode.Children.Count; ++i)
            {
                prediction = nn.Predict(MCTSRootNode.Children[i].Value);
                MCTSRootNode.Children[i].nn_policy = new List<float>(prediction.Item1);
                MCTSRootNode.Children[i].nn_value = prediction.Item2;
            }

            for (int i = 0; i < 5; ++i)
            {
                Console.WriteLine(MCTSRootNode.Children[i * 5 + 0].nn_value.ToString("0.000") + " " +
                    MCTSRootNode.Children[i * 5 + 1].nn_value.ToString("0.000") + " " +
                    MCTSRootNode.Children[i * 5 + 2].nn_value.ToString("0.000") + " " +
                    MCTSRootNode.Children[i * 5 + 3].nn_value.ToString("0.000") + " " +
                    MCTSRootNode.Children[i * 5 + 4].nn_value.ToString("0.000") + " ");
            }
            Console.WriteLine("\n");
        }
        private float getNoiseWeight(int depth)
        {
            float noiseWeight = 0.0f;
            if (Params.DN_SCALING == DIRICHLET_NOISE_SCALING.CONSTANT)
            {
                noiseWeight = Params.DIRICHLET_NOISE_WEIGHT; // constant
            }
            else if (Params.DN_SCALING == DIRICHLET_NOISE_SCALING.LINEAR)
            {
                noiseWeight = ((25 - depth) / 25.0f) * Params.DIRICHLET_NOISE_WEIGHT;
            }
            else if (Params.DN_SCALING == DIRICHLET_NOISE_SCALING.QUADRATIC)
            {
                noiseWeight = ((25 - depth) / 25.0f) * ((25 - depth) / 25.0f) * Params.DIRICHLET_NOISE_WEIGHT;
            }
            else if (Params.DN_SCALING == DIRICHLET_NOISE_SCALING.FIRST_NODE_ONLY)
            {
                if (depth == 0)
                {
                    noiseWeight = Params.DIRICHLET_NOISE_WEIGHT;
                }
                else
                {
                    noiseWeight = 0.0f;
                }
            }
            return noiseWeight;
        }
        private int findBestChildWinrate(Node<TicTacToePosition> currNode, DirichletNoise dn, int depth)
        {
            float best_winrate = float.NegativeInfinity;
            int best_child_index = -1;

            /* add dirichlet noise to root */
            for (int i = 0; i < currNode.Children.Count; ++i)
            {
                float noiseWeight = getNoiseWeight(depth);
               
                float winrate_temp = currNode.Children[i].winrate * (1 - noiseWeight) + noiseWeight * dn.GetNoise(i);
                if (winrate_temp > best_winrate)
                {
                    best_winrate = winrate_temp;
                    best_child_index = i;
                }
            }

            return best_child_index;
        }
        private int findBestChildVisitCount(Node<TicTacToePosition> currNode, DirichletNoise dn, int depth)
        {
            float best_visit_count = 0;
            int best_child_index = -1;

            for (int i = 0; i < currNode.Children.Count; ++i)
            {
                float noiseWeight = getNoiseWeight(depth);

                float tempvisitCount = currNode.Children[i].visitCount * (1 - noiseWeight) + noiseWeight * dn.GetNoise(i);
                if (tempvisitCount > best_visit_count)
                {
                    best_visit_count = tempvisitCount;
                    best_child_index = i;
                }
            }
            return best_child_index;
        }
        private void createChildren(Node<TicTacToePosition> currNode)
        {
            if (!currNode.HasChild)
            {
                TicTacToeGame game = new TicTacToeGame(currNode.Value);
                List<Tuple<int, int>> moves = game.GetMoves(); // valid moves

                for (int i = 0; i < moves.Count; ++i)
                {
                    game = new TicTacToeGame(currNode.Value);
                    game.DoMove(moves[i]);
                    Node<TicTacToePosition> child = new Node<TicTacToePosition>(currNode);
                    child.Value = new TicTacToePosition(game.position);
                    child.moveIndex = moves[i].Item1 * 5 + moves[i].Item2;
                    currNode.AddChild(child);
                }
            }
        }
        private void calculateNNOutput(Node<TicTacToePosition> currNode, NeuralNetwork NN)
        {
            Tuple<float[], float> prediction = NN.Predict(currNode.Value);
            currNode.nn_policy = new List<float>(prediction.Item1);
            currNode.nn_value = prediction.Item2;
        }
        /// <summary>
        /// Sends prediction request to neural network and adds the node to a queue to later fill in the result
        /// </summary>
        /// <param name="currNode"></param>
        /// <param name="NN"></param>
        /// <param name="queue"></param>
        private void calculateNNOutputGPU(Node<TicTacToePosition> currNode, NeuralNetwork NN, Queue<Node<TicTacToePosition>> queue)
        {
            currNode.waitingForGPUPrediction = true;
            propagateVirtualLoss(currNode);
            NN.PredictGPU(currNode.Value);
            queue.Enqueue(currNode);
        }
        private Node<TicTacToePosition> findMostPromisingLeafNode(Node<TicTacToePosition> currNode)
        {
            while (currNode.HasChild)
            {
                /* create the game from the TicTacToePosition */
                TicTacToeGame game = new TicTacToeGame(currNode.Value);
                List<Tuple<int, int>> moves = game.GetMoves(); // valid moves

                /* find best child node (best UCT value) to expand */
                float bestUCTScore = float.NegativeInfinity;
                int bestChildIndex = -1;

                //List<float> winratesChildren = new List<float>(); // we want to check if all children have 0 winrates and policy doesnt exist, because then it is better 
                // to choose a random child not the last one

                // if nnpolicy is null then also all children have no nn output, but possibly a score from endgame position
                for (int i = 0; i < currNode.Children.Count; ++i)
                {
                    // this factors in all virtual losses into the winrate
                    float childWinrateWithVirtualLoss = (currNode.Children[i].winrate * currNode.Children[i].visitCount) /
                        (currNode.Children[i].visitCount + currNode.Children[i].virtualLossCount+1);
                    //winratesChildren.Add(childWinrateWithVirtualLoss);

                    float temp_UCT_score = childWinrateWithVirtualLoss;
                    if (currNode.nn_policy != null)
                    {
                        temp_UCT_score = childWinrateWithVirtualLoss + Params.C_PUCT * currNode.nn_policy[currNode.Children[i].moveIndex] *
                            (float)Math.Sqrt(currNode.visitCount) / (float)(currNode.Children[i].visitCount + 1);
                    }
                    else
                    {
                        // assume policy equal for all children if not found yet
                        temp_UCT_score = childWinrateWithVirtualLoss + Params.C_PUCT * (1.0f/currNode.Children.Count) *
                            (float)Math.Sqrt(currNode.visitCount) / (float)(currNode.Children[i].visitCount + 1);
                    }
                    if (temp_UCT_score > bestUCTScore)
                    {
                        // new best child 
                        bestChildIndex = i;
                        bestUCTScore = temp_UCT_score;
                    }
                }

                currNode = currNode.Children[bestChildIndex];
            }
            return currNode;
        }
        private void backpropagateScore(Node<TicTacToePosition> currNode, float score)
        {
            // we store the winrate for the opposite player in the node, during search we look at the next level
            while (currNode != null)
            {
                if (currNode.Value.sideToMove == Player.X)
                {
                    currNode.winrate = (currNode.visitCount * currNode.winrate + 1.0f - score) / (currNode.visitCount + 1);
                }
                else if (currNode.Value.sideToMove == Player.Z)
                {
                    currNode.winrate = (currNode.visitCount * currNode.winrate + score) / (currNode.visitCount + 1);
                }
                currNode.visitCount += 1;
                currNode = currNode.GetParent();
            }
        }
        private void RegularMCTSSearch(Node<TicTacToePosition> currNode)
        {
            TicTacToeGame game = new TicTacToeGame(currNode.Value);
            List<Tuple<int, int>> moves = game.GetMoves();

            /* find the most promising leaf node */
            currNode = findMostPromisingLeafNode(currNode);

            /* if the leaf node is a game ending state use correct score */
            float score = 0.0f;
            game = new TicTacToeGame(currNode.Value);
            if (game.IsOver())
            {
                score = (game.GetScore() + 1.0f)/ 2.0f;
            }
            else
            {
                moves = game.GetMoves();

                /* create children of normal leaf */
                createChildren(currNode);

                /* choose random child.. */
                int best_policy_child_index = RandomGen2.Next(0, currNode.Children.Count);

                /*get the value of best child..*/
                currNode = currNode.Children[best_policy_child_index];

                score = (simulateRandomPlayout(currNode) + 1.0f)/ 2.0f; // [-1..1] where player X wins at 1 and player Z wins at -1
            }

            /* update the tree with the new score and visit counts */
            backpropagateScore(currNode, score);
        }

        /// <summary>
        /// Play randomly until game is over and update all winrates in the tree
        /// </summary>
        /// <param name="currNode"></param>
        /// <returns></returns>
        private float simulateRandomPlayout(Node<TicTacToePosition> currNode)
        {
            TicTacToeGame game = new TicTacToeGame(currNode.Value);

            while (!game.IsOver())
            {
                List<Tuple<int, int>> moves = game.GetMoves();

                game.DoMove(moves[RandomGen2.Next(0, moves.Count)]);
            }

            return game.GetScore();
        }

        private void DisplayMCTSTree(Node<TicTacToePosition> roodNode)
        {
            Console.WriteLine("============================ Displaying MCTS Tree (DF order) ============================\n");
            DisplayMCTSNode(roodNode);
        }
        private void DisplayMCTSNode(Node<TicTacToePosition> node)
        {
            if (node.visitCount != 0)
            {
                Console.WriteLine(node.ToString());
                foreach (Node<TicTacToePosition> child in node.Children)
                {
                    DisplayMCTSNode(child);
                }
            }
        }
    }
}
