using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Threading;
using System.Diagnostics;
using System.Reflection;

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
                OpenCL.Init(Math.Max(Params.NOF_CPU_THREADS_GPU_WORKLOAD*2, Math.Max(Params.NOF_GAMES_TEST*2, Params.NOF_OFFSPRING * 2)));
        }
        /// <summary>
        /// Main loop for neuroevolution training
        /// </summary>
        public void Train()
        {
            bestNN = new NeuralNetwork(currentNN.weights, currentNN.untrainable_weights);

            currentNN.SaveWeightsToFile("weights_neuro_start.txt");

            for (int i = 0; i < Params.NOF_EPOCHS; ++i)
            {
                TrainingRun(i);
                if (i % Params.SAVE_WEIGHT_EVERY_XTH_EPOCH == 0)
                    currentNN.SaveWeightsToFile("weights_net_" + ((int)(i / Params.SAVE_WEIGHT_EVERY_XTH_EPOCH)).ToString() + ".txt");
            }
        }
        /// <summary>
        /// Main loop for Backpropagation training
        /// </summary>
        public void TrainKeras()
        {
            bestNN = new NeuralNetwork(currentNN.weights);
            printNNOutput(bestNN);

            // the old network
            Params.DIRICHLET_NOISE_WEIGHT = 0.2f;
            //CheckPerformanceVsRandomKeras(bestNN, Params.NOF_GAMES_VS_RANDOM);

            WritePlotStatistics();

            while (true)
            {
                Console.WriteLine("Main Thread: Epoch start");

                Params.DIRICHLET_NOISE_WEIGHT = 0.2f;
                String filename = ProduceTrainingGamesKeras(bestNN, Params.NOF_GAMES_TRAIN_KERAS);

                ProcessStartInfo pythonInfo = new ProcessStartInfo();
                Process python;
                pythonInfo.FileName = @"python.exe";//@"C:\Users\Admin\Anaconda3\envs\NALU\python.exe";
                pythonInfo.Arguments = "\"Z:\\CloudStation\\GitHub Projects\\TicTacToe-DL-RL\\Training\\main.py \" " + filename; // TODO: should be relative
                pythonInfo.CreateNoWindow = false;
                pythonInfo.UseShellExecute = false;

                var location = new Uri(Assembly.GetEntryAssembly().GetName().CodeBase);
                String exePath = new FileInfo(location.AbsolutePath).Directory.FullName;

                pythonInfo.RedirectStandardOutput = true;

                Console.WriteLine("Invoking python script...");
                python = Process.Start(pythonInfo);

                while (!python.StandardOutput.EndOfStream)
                {
                    string line = python.StandardOutput.ReadLine();
                    Console.WriteLine(line);
                }

                python.WaitForExit();
                python.Close();

                // #################################### TEST NEW NETWORK ##########################################

                currentNN.ReadWeightsFromFileKeras("./../../../Training/weights.txt"); // must have been created with python script
                Params.DIRICHLET_NOISE_WEIGHT = 0.0f;
                bool newBestFound = CheckPerformanceVsOldNet(currentNN, bestNN, Params.NOF_GAMES_TEST);

                // #################################### CREATE NEW BEST NETWORK ##########################################

                if (!newBestFound)
                {
                    printNNOutput(bestNN);
                }
                else
                { 
                    Console.WriteLine("New best network found!");
                    printNNOutput(currentNN);

                    bestNN = new NeuralNetwork(currentNN.weights);

                    Params.DIRICHLET_NOISE_WEIGHT = 0.2f;
                    CheckPerformanceVsRandomKeras(bestNN, Params.NOF_GAMES_VS_RANDOM);
                }

                WritePlotStatistics();
            }
        }
        public void ValidateOuputGPU()
        {
            ID.ResetGlobalID();
            OpenCL.ClearWeights();

            NeuralNetwork nn1 = new NeuralNetwork(currentNN.weights);
            NeuralNetwork nn2 = new NeuralNetwork(currentNN.weights);
            for (int i = 0; i < 1; ++i)
            {
                nn1.SetIds(i,i);
                nn1.ChannelInit();
                nn1.EnqueueWeights();
                nn2.SetIds(i,i);
                nn2.ChannelInit();
                nn2.EnqueueWeights();
            }
            OpenCL.CreateNetworkWeightBuffers();
            Thread thread = new Thread(OpenCL.Run);
            //thread.Priority = ThreadPriority.Highest;
            thread.Start();

            for (int z = 0; z < 10; ++z)
            {
                TicTacToeGame game = new TicTacToeGame();
                game.DoMove(Tuple.Create(3, 2));
                game.DoMove(Tuple.Create(0, 1));
                game.DoMove(Tuple.Create(0, 4)); // y, x
                game.DoMove(Tuple.Create(2, 4));

                nn1.PredictGPU(game.position);
                nn2.PredictGPU(game.position);


                // if we need to wait then wait
                Tuple<float[], float> prediction1 = nn1.GetResultSync();
                Tuple<float[], float> prediction2 = nn2.GetResultSync();

                for (int i = 0; i < 5; ++i)
                {
                    Console.WriteLine(prediction1.Item1[i * 5 + 0].ToString("0.000") + " " +
                    prediction1.Item1[i * 5 + 1].ToString("0.000") + " " +
                    prediction1.Item1[i * 5 + 2].ToString("0.000") + " " +
                    prediction1.Item1[i * 5 + 3].ToString("0.000") + " " +
                    prediction1.Item1[i * 5 + 4].ToString("0.000"));
                }
                Console.WriteLine("Value " + prediction1.Item2);
                Console.WriteLine("\n");

                for (int i = 0; i < 5; ++i)
                {
                    Console.WriteLine(prediction2.Item1[i * 5 + 0].ToString("0.000") + " " +
                    prediction2.Item1[i * 5 + 1].ToString("0.000") + " " +
                    prediction2.Item1[i * 5 + 2].ToString("0.000") + " " +
                    prediction2.Item1[i * 5 + 3].ToString("0.000") + " " +
                    prediction2.Item1[i * 5 + 4].ToString("0.000"));
                }
                Console.WriteLine("Value " + prediction2.Item2);
                Console.WriteLine("\n");
            }
            thread.Abort();
            ValidateOutputCPU();
        }
        public void ValidateOutputCPU()
        {

            TicTacToeGame game = new TicTacToeGame();
            game.DoMove(Tuple.Create(3, 2));
            game.DoMove(Tuple.Create(0, 1));
            game.DoMove(Tuple.Create(0, 4)); // y, x
            game.DoMove(Tuple.Create(2, 4));

            Tuple<float[], float> prediction = currentNN.Predict(game.position);

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
        /// <summary>
        /// Keras only, returns whether network should be replaced
        /// </summary>
        /// <param name="nofGames"></param>
        /// <param name="newNN"></param>
        /// <param name="oldNN"></param>
        public bool CheckPerformanceVsOldNet(NeuralNetwork newNN, NeuralNetwork oldNN, int nofGames)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();
            Console.WriteLine("Main Thread: CPU test games starting...");

            List<int> wins = new List<int>(new int[nofGames]);
            List<int> draws = new List<int>(new int[nofGames]);
            List<int> losses = new List<int>(new int[nofGames]);
            List<int> movecount = new List<int>(new int[nofGames]);
            List<int> winsX = new List<int>(new int[nofGames]);
            List<int> winsZ = new List<int>(new int[nofGames]);

            List<NeuralNetwork> nns = new List<NeuralNetwork>();
            List<NeuralNetwork> currnns = new List<NeuralNetwork>();


            // ################################# COPY WEIGHTS TO GPU MEMORY ###########################################

            if (Params.GPU_ENABLED)
            {
                ID.ResetGlobalID();
                OpenCL.ClearWeights();

                NeuralNetwork nn1 = new NeuralNetwork(newNN.weights);
                NeuralNetwork nn2 = new NeuralNetwork(oldNN.weights);
                nn1.SetIds(0, 0);
                nn2.SetIds(1, 1);

                nn2.EnqueueWeights();
                nn1.EnqueueWeights();
                nn1.ChannelInit();
                nn2.ChannelInit();

                nns.Add(nn2);
                currnns.Add(nn1);

                for (int i = 1; i < nofGames; ++i)
                {
                    nn1 = new NeuralNetwork();
                    nn1.DeleteArrays();
                    nn1.SetIds(0, i*2);
                    nn1.ChannelInit();

                    nn2 = new NeuralNetwork();
                    nn2.DeleteArrays();
                    nn2.SetIds(1, i*2+1);
                    nn2.ChannelInit();

                    nns.Add(nn2);
                    currnns.Add(nn1);
                }

                OpenCL.CreateNetworkWeightBuffers();
            }

            // ###################################### GPU LOOP ##############################################

            if (Params.GPU_ENABLED)
            {
                Thread thread = new Thread(OpenCL.Run);
                thread.Priority = ThreadPriority.Highest;
                thread.Start();

                using (var progress = new ProgressBar())
                {
                    long sharedLoopCounter = 0;

                    ThreadPool.SetMinThreads(nofGames, nofGames);
                    // process batches of games to re-use neural networks
                    Parallel.For(0, nofGames,
                        new ParallelOptions { MaxDegreeOfParallelism = nofGames }, i =>
                        {
                            List<Tuple<int, int>> history = new List<Tuple<int, int>>();
                            Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;

                            int result = PlayOneGameGPU(history, evaluationNetworkPlayer, 
                                currnns[i], nns[i], false);

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

                            /* to display some games */
                            if (i >= nofGames - 2)
                            {
                                if ((i % 2) == 0)
                                    Console.WriteLine("Eval player playing as Player X");
                                else
                                    Console.WriteLine("Eval player playing as Player Z");
                                TicTacToeGame game = new TicTacToeGame();
                                game.DisplayHistory(history);
                            }
                            Interlocked.Add(ref sharedLoopCounter, 1);
                            progress.Report((double)Interlocked.Read(ref sharedLoopCounter) / nofGames);
                        });
                    }
                    thread.Abort();
            }
            else
            {
                for (int i = 0; i < nofGames; ++i)
                {
                    nns.Add(new NeuralNetwork(oldNN.weights));
                    currnns.Add(new NeuralNetwork(newNN.weights));
                }

                using (var progress = new ProgressBar())
                {
                    long sharedLoopCounter = 0;

                    Parallel.For(0, nofGames,
                        new ParallelOptions { MaxDegreeOfParallelism = Params.NOF_CPU_THREADS_CPU_WORKLOAD }, i =>
                        {
                            List<Tuple<int, int>> history = new List<Tuple<int, int>>();
                            Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;

                            int result = PlayOneGame(history, evaluationNetworkPlayer, currnns[i], 
                                nns[i], false);

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

                            /* to display some games*/
                            if (i >= nofGames - 2)
                            {
                                if ((i % 2) == 0)
                                    Console.WriteLine("Eval player playing as Player X");
                                else
                                    Console.WriteLine("Eval player playing as Player Z");
                                TicTacToeGame game = new TicTacToeGame();
                                game.DisplayHistory(history);
                            }
                            Interlocked.Add(ref sharedLoopCounter, 1);
                            progress.Report((double)Interlocked.Read(ref sharedLoopCounter) / nofGames);
                    });
                }
            }

            bool plsReplaceMe = false;
            float totalWins = wins.Sum();
            float totalWinsAsX = winsX.Sum();
            float totalWinsAsZ = winsZ.Sum();
            float totalDraws = draws.Sum();
            float totalMoves = movecount.Sum();
            drawsMovingAvg.ComputeAverage((Decimal)totalDraws/nofGames);
            winsAsXMovingAvg.ComputeAverage((Decimal)totalWinsAsX/nofGames);
            winsAsZMovingAvg.ComputeAverage((Decimal)totalWinsAsZ/nofGames);
            averageMovesMovingAvg.ComputeAverage((Decimal)totalMoves/nofGames);

            if (((totalWins + 0.5f * totalDraws) / nofGames) * 100.0f >= Params.MINIMUM_WIN_PERCENTAGE)
            {
                plsReplaceMe = true;

                currentPseudoELO += (float)(totalWins - (nofGames-totalDraws-totalWins)) / (float)nofGames;
            }

            sw.Stop();
            Console.WriteLine("Main Thread: Vs. previous best: W/D/L : " + totalWins + " " + totalDraws + " " + (nofGames - totalDraws - totalWins) + " - " +
                Math.Round((((totalWins + totalDraws * 0.5) / nofGames) * 100.0f), 2) + "%");
            Console.WriteLine("Main Thread: Finished in: " + sw.ElapsedMilliseconds + "ms");

            return plsReplaceMe;
        }
        public void CheckPerformanceVsRandomKeras(NeuralNetwork nn, int nofGames)
        {
            List<NeuralNetwork> currnns = new List<NeuralNetwork>();

            for (int i = 0; i < nofGames; ++i)
            {
                currnns.Add(new NeuralNetwork(nn.weights));
            }

            Console.WriteLine("Main Thread: CPU run vs random player...");
            List<float> winsVsRand = new List<float>(new float[nofGames]);

            using (var progress = new ProgressBar())
            {
                long sharedLoopCounter = 0;
                Parallel.For(0, nofGames, new ParallelOptions { MaxDegreeOfParallelism = Params.NOF_CPU_THREADS_CPU_WORKLOAD }, i =>
                {
                    Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;
                    winsVsRand[i] += PlayAgainstRandom(currnns[i], evaluationNetworkPlayer, Params.NOF_SIMS_PER_MOVE_VS_RANDOM1);
                    Interlocked.Add(ref sharedLoopCounter, 1);
                    progress.Report((double)Interlocked.Read(ref sharedLoopCounter) / nofGames);
                });
            }

            winrateVsRandTotal1 = (float)winsVsRand.Average();
            winrateVsRandMovingAvg1.ComputeAverage((decimal)winrateVsRandTotal1);
            Console.WriteLine("Main Thread: Wins/Games vs Random Player (" + Params.NOF_SIMS_PER_MOVE_VS_RANDOM1 + " Nodes): " + Math.Round(winrateVsRandTotal1 * 100, 2) + "%");

            winsVsRand = new List<float>(new float[nofGames]);
            using (var progress = new ProgressBar())
            {
                long sharedLoopCounter = 0;
                Parallel.For(0, nofGames, new ParallelOptions { MaxDegreeOfParallelism = Params.NOF_CPU_THREADS_CPU_WORKLOAD }, i =>
                {
                    Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;
                    winsVsRand[i] += PlayAgainstRandom(currnns[i], evaluationNetworkPlayer, Params.NOF_SIMS_PER_MOVE_VS_RANDOM2);
                    Interlocked.Add(ref sharedLoopCounter, 1);
                    progress.Report((double)Interlocked.Read(ref sharedLoopCounter) / nofGames);
                });
            }

            winrateVsRandTotal2 = (float)winsVsRand.Average();
            winrateVsRandMovingAvg2.ComputeAverage((decimal)winrateVsRandTotal2);
            Console.WriteLine("Main Thread: Wins/Games vs Random Player (" + Params.NOF_SIMS_PER_MOVE_VS_RANDOM2 + " Nodes): " + Math.Round(winrateVsRandTotal2 * 100, 2) + "%");

            winsVsRand = new List<float>(new float[nofGames]);
            using (var progress = new ProgressBar())
            {
                long sharedLoopCounter = 0;
                Parallel.For(0, nofGames, new ParallelOptions { MaxDegreeOfParallelism = Params.NOF_CPU_THREADS_CPU_WORKLOAD }, i =>
                {
                    Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;
                    winsVsRand[i] += PlayAgainstRandom(currnns[i], evaluationNetworkPlayer, Params.NOF_SIMS_PER_MOVE_VS_RANDOM3);
                    Interlocked.Add(ref sharedLoopCounter, 1);
                    progress.Report((double)Interlocked.Read(ref sharedLoopCounter) / nofGames);
                });
            }
            winrateVsRandTotal3 = (float)winsVsRand.Average();
            winrateVsRandMovingAvg3.ComputeAverage((decimal)winrateVsRandTotal3);
            Console.WriteLine("Main Thread: Wins/Games vs Random Player (" + Params.NOF_SIMS_PER_MOVE_VS_RANDOM3 + " Nodes): " + Math.Round(winrateVsRandTotal3 * 100, 2) + "%");
        }

        public void WritePlotStatistics()
        {
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
        /// Return file name for games
        /// </summary>
        /// <param name="nofGames"></param>
        /// <returns></returns>
        public String ProduceTrainingGamesKeras(NeuralNetwork nn, int nofGames)
        {
            Console.WriteLine("Main Thread: Creating " + nofGames + " training games...");

            List<NeuralNetwork> nns = new List<NeuralNetwork>();
            List<NeuralNetwork> currnns = new List<NeuralNetwork>();
            List<Node<TicTacToePosition>> rootNodes = new List<Node<TicTacToePosition>>();

            if (!Params.GPU_ENABLED)
            {
                for (int i = 0; i < Params.NOF_CPU_THREADS_GPU_WORKLOAD; ++i)
                {
                    NeuralNetwork playingNNlocal = new NeuralNetwork(nn.weights);
                    nns.Add(playingNNlocal);

                    NeuralNetwork currNNlocal = new NeuralNetwork(nn.weights);
                    currnns.Add(currNNlocal);
                }
            }

            // ################################# COPY WEIGHTS TO GPU MEMORY ###########################################

            if (Params.GPU_ENABLED)
            {
                ID.ResetGlobalID();
                OpenCL.ClearWeights();

                for (int i = 0; i < Params.NOF_CPU_THREADS_GPU_WORKLOAD; ++i)
                {
                    NeuralNetwork playingNNlocal = new NeuralNetwork();
                    playingNNlocal.DeleteArrays();
                    playingNNlocal.SetIds(0, ID.GetGlobalID());
                    playingNNlocal.ChannelInit();

                    nns.Add(playingNNlocal);
                    rootNodes.Add(new Node<TicTacToePosition>(null));
                }

                nns[0] = new NeuralNetwork(nn.weights);
                nns[0].SetIds(0, 0);
                nns[0].ChannelInit();
                nns[0].EnqueueWeights();
                OpenCL.CreateNetworkWeightBuffers();
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

            // ###################################### GPU LOOP ##############################################

            if (Params.GPU_ENABLED)
            {
                Thread thread = new Thread(OpenCL.Run);
                thread.Priority = ThreadPriority.Highest;
                thread.Start();

                using (var progress = new ProgressBar())
                {
                    long sharedLoopCounter = 0;
                    ThreadPool.SetMinThreads(Params.NOF_CPU_THREADS_GPU_WORKLOAD, Params.NOF_CPU_THREADS_GPU_WORKLOAD);
                    for (int j = 0; j < nofGames / Params.NOF_CPU_THREADS_GPU_WORKLOAD; ++j)
                    { // process batches of games to re-use neural networks
                        Parallel.For(j* Params.NOF_CPU_THREADS_GPU_WORKLOAD, j * Params.NOF_CPU_THREADS_GPU_WORKLOAD + Params.NOF_CPU_THREADS_GPU_WORKLOAD, 
                            new ParallelOptions { MaxDegreeOfParallelism = Params.NOF_CPU_THREADS_GPU_WORKLOAD }, i =>
                        {
                            Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z; // doesnt really matter for 2 equal networks
                            scores[i] = RecordOneGameGPU(moves[i], policies[i], evaluationNetworkPlayer,
                                nns[i % Params.NOF_CPU_THREADS_GPU_WORKLOAD], rootNodes[i % Params.NOF_CPU_THREADS_GPU_WORKLOAD], true);
                            Interlocked.Add(ref sharedLoopCounter, 1);
                            progress.Report((double)Interlocked.Read(ref sharedLoopCounter) / nofGames);
                        });
                    }
                }
                thread.Abort();
            }

            // ###################################### CPU LOOP ##############################################
            else
            {
                using (var progress = new ProgressBar())
                {
                    long sharedLoopCounter = 0;
                    ThreadPool.SetMinThreads(Params.NOF_CPU_THREADS_CPU_WORKLOAD, Params.NOF_CPU_THREADS_CPU_WORKLOAD);
                    for (int j = 0; j < nofGames / Params.NOF_CPU_THREADS_CPU_WORKLOAD; ++j)
                    { // process batches of games to re-use neural networks
                        Parallel.For(j * Params.NOF_CPU_THREADS_CPU_WORKLOAD, j *Params.NOF_CPU_THREADS_CPU_WORKLOAD + Params.NOF_CPU_THREADS_CPU_WORKLOAD,
                            new ParallelOptions { MaxDegreeOfParallelism = Params.NOF_CPU_THREADS_CPU_WORKLOAD }, i =>
                            {
                                Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z; // doesnt really matter for 2 equal networks
                                scores[i] = RecordOneGame(moves[i], policies[i], evaluationNetworkPlayer, 
                                    currnns[i % Params.NOF_CPU_THREADS_CPU_WORKLOAD], nns[i % Params.NOF_CPU_THREADS_CPU_WORKLOAD], true);
                                Interlocked.Add(ref sharedLoopCounter, 1);
                                progress.Report((double)Interlocked.Read(ref sharedLoopCounter) / nofGames);
                            });
                    }
                }
            }
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
                    nns[i].SetIds(i*2, i * 2);
                    currnns[i].SetIds(i*2+1, i * 2+1);

                    nns[i].EnqueueWeights();
                    currnns[i].EnqueueWeights();
                    nns[i].ChannelInit();
                    currnns[i].ChannelInit();
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
                Thread thread = new Thread(OpenCL.Run);
                thread.Priority = ThreadPriority.Highest;
                thread.Start();

                int numOfThreads = Math.Min(Params.NOF_OFFSPRING, Params.NOF_CPU_THREADS_GPU_WORKLOAD);
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
                using (var progress = new ProgressBar())
                {
                    long sharedLoopCounter = 0;
                    Parallel.For(0, Params.NOF_OFFSPRING, new ParallelOptions { MaxDegreeOfParallelism = Params.NOF_CPU_THREADS_GPU_WORKLOAD }, i =>
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
                        Interlocked.Add(ref sharedLoopCounter, 1);
                        progress.Report((double)Interlocked.Read(ref sharedLoopCounter) / Params.NOF_OFFSPRING);
                    });
                }
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
                    nns[i].SetIds(i*2, ID.GetGlobalID());
                    currnns[i].SetIds(i * 2+1, ID.GetGlobalID());
                    nns[i].ChannelInit();
                    currnns[i].ChannelInit();
                    nns[i].EnqueueWeights();
                    currnns[i].EnqueueWeights();
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

                int numOfThreads = Math.Min(Params.NOF_GAMES_TEST, Params.NOF_CPU_THREADS_GPU_WORKLOAD);
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
                            if (jk % 2 == 0)
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

                using (var progress = new ProgressBar())
                {
                    long sharedLoopCounter = 0;
                    Parallel.For(0, Params.NOF_GAMES_TEST, new ParallelOptions { MaxDegreeOfParallelism = Params.NOF_CPU_THREADS_GPU_WORKLOAD }, i =>
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
                            if ((i % 2) == 0)
                                Console.WriteLine("Eval player playing as Player X");
                            else
                                Console.WriteLine("Eval player playing as Player Z");
                            TicTacToeGame game = new TicTacToeGame();
                            game.DisplayHistory(history);
                        }
                        Interlocked.Add(ref sharedLoopCounter, 1);
                        progress.Report((double)Interlocked.Read(ref sharedLoopCounter) / Params.NOF_GAMES_TEST);
                    });
                }
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
            if (((winsTotal+drawsTotal*0.5f)/ Params.NOF_GAMES_TEST)*100.0f < Params.MINIMUM_WIN_PERCENTAGE)
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

                printNNOutput(bestNN);
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
                Parallel.For(0, Params.NOF_GAMES_VS_RANDOM, new ParallelOptions { MaxDegreeOfParallelism = Params.NOF_CPU_THREADS_GPU_WORKLOAD }, i =>
                {
                    Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;
                    winsVsRand[i] += PlayAgainstRandom(currnns[i], evaluationNetworkPlayer, Params.NOF_SIMS_PER_MOVE_VS_RANDOM1);
                });
                winrateVsRandTotal1 = (float)winsVsRand.Average();
                winrateVsRandMovingAvg1.ComputeAverage((decimal)winrateVsRandTotal1);
                Console.WriteLine("Main Thread: Wins/Games vs Random Player (" + Params.NOF_SIMS_PER_MOVE_VS_RANDOM1 + " Nodes): " + Math.Round(winrateVsRandTotal1 * 100,2) + "%");

                winsVsRand = new List<float>(new float[Params.NOF_GAMES_VS_RANDOM]);
                Parallel.For(0, Params.NOF_GAMES_VS_RANDOM, new ParallelOptions { MaxDegreeOfParallelism = Params.NOF_CPU_THREADS_GPU_WORKLOAD }, i =>
                {
                    Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;
                    winsVsRand[i] += PlayAgainstRandom(currnns[i], evaluationNetworkPlayer, Params.NOF_SIMS_PER_MOVE_VS_RANDOM2);
                });
                winrateVsRandTotal2 = (float)winsVsRand.Average();
                winrateVsRandMovingAvg2.ComputeAverage((decimal)winrateVsRandTotal2);
                Console.WriteLine("Main Thread: Wins/Games vs Random Player (" + Params.NOF_SIMS_PER_MOVE_VS_RANDOM2 + " Nodes): " + Math.Round(winrateVsRandTotal2 * 100,2) + "%");

                winsVsRand = new List<float>(new float[Params.NOF_GAMES_VS_RANDOM]);
                Parallel.For(0, Params.NOF_GAMES_VS_RANDOM, new ParallelOptions { MaxDegreeOfParallelism = Params.NOF_CPU_THREADS_GPU_WORKLOAD }, i =>
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

            WritePlotStatistics();
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

                int best_child_index = -1;
                if (game.position.sideToMove == evaluationNetworkPlayer)
                {
                    for (int simulation = 0; simulation < nofSimsPerMove; ++simulation)
                    {
                        SearchUsingNN(MCTSRootNode, NN);
                    }
                    best_child_index = findBestChildVisitCount(MCTSRootNode);
                }
                else
                {
                    createChildren(MCTSRootNode);
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

                createChildren(MCTSRootNodeNN1);
                /* find value, policy */
                if (MCTSRootNodeNN1.nn_policy == null)
                {
                    calculateNNOutput(MCTSRootNodeNN1, NN1, true);
                    
                    backpropagateScore(MCTSRootNodeNN1, MCTSRootNodeNN1.nn_value);
                }
                createChildren(MCTSRootNodeNN2);
                if (MCTSRootNodeNN2.nn_policy == null)
                {
                    calculateNNOutput(MCTSRootNodeNN2, NN2, true);
                    backpropagateScore(MCTSRootNodeNN2, MCTSRootNodeNN2.nn_value);
                }
                int nofSimsPerMove = train ? Params.NOF_SIMS_PER_MOVE_TRAINING : Params.NOF_SIMS_PER_MOVE_TESTING;

                for (int simulation = 0; simulation < nofSimsPerMove; ++simulation)
                {
                    if (curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                        SearchUsingNN(MCTSRootNodeNN1, NN1); // if its the turn of eval player
                    else
                    {
                        SearchUsingNN(MCTSRootNodeNN2, NN2); // expand tree and improve accuracy at MCTSRootNode
                    }

                    // show last simulation tree
                    if (simulation == nofSimsPerMove - 1 && curr_ply == 0)
                    {
                        //DisplayMCTSTree(MCTSRootNode);
                    }
                }

                int best_child_index = -1;
                if (curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                    if (train)
                        best_child_index = findBestChildVisitCountStochastic(MCTSRootNodeNN1);
                    else
                        best_child_index = findBestChildVisitCount(MCTSRootNodeNN1);
                else
                    if (train)
                        best_child_index = findBestChildVisitCountStochastic(MCTSRootNodeNN2);
                    else
                        best_child_index = findBestChildVisitCount(MCTSRootNodeNN2);

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

                createChildren(MCTSRootNodeNN1);
                /* find value, policy */
                if (MCTSRootNodeNN1.nn_policy == null)
                {
                    calculateNNOutput(MCTSRootNodeNN1, NN1, true); // ignore root value
                    backpropagateScore(MCTSRootNodeNN1, MCTSRootNodeNN1.nn_value);
                }
                createChildren(MCTSRootNodeNN2);
                if (MCTSRootNodeNN2.nn_policy == null)
                {
                    calculateNNOutput(MCTSRootNodeNN2, NN2, true);
                    backpropagateScore(MCTSRootNodeNN2, MCTSRootNodeNN2.nn_value);
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
                        //DisplayMCTSTree(MCTSRootNodeNN1);
                        //DisplayMCTSTree(MCTSRootNodeNN2);
                    }
                }

                // after search, record root node visits for new policy vector
                if (curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                {
                    List<float> policy = Enumerable.Repeat(0.0f, 25).ToList();
                    float totalVisits = 0;
                    for (int i = 0; i < MCTSRootNodeNN1.Children.Count; ++i)
                    {
                        policy[MCTSRootNodeNN1.Children[i].moveIndex] = MCTSRootNodeNN1.Children[i].visits;
                        totalVisits += MCTSRootNodeNN1.Children[i].visits;
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
                        policy[MCTSRootNodeNN2.Children[i].moveIndex] = MCTSRootNodeNN2.Children[i].visits;
                        totalVisits += MCTSRootNodeNN2.Children[i].visits;
                    }
                    for (int i = 0; i < MCTSRootNodeNN2.Children.Count; ++i)
                    {
                        policy[MCTSRootNodeNN2.Children[i].moveIndex] /= totalVisits;
                        if (policy[MCTSRootNodeNN2.Children[i].moveIndex] == float.NaN ||
                            policy[MCTSRootNodeNN2.Children[i].moveIndex] == float.NegativeInfinity ||
                            policy[MCTSRootNodeNN2.Children[i].moveIndex] == float.PositiveInfinity)
                        {
                            policy[MCTSRootNodeNN2.Children[i].moveIndex] = 0.0f;
                        }
                    }

                    policies.Add(policy);
                }

                int best_child_index = -1;
                if (curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                    if (train)
                        best_child_index = findBestChildVisitCountStochastic(MCTSRootNodeNN1);
                    else
                        best_child_index = findBestChildVisitCount(MCTSRootNodeNN1);
                else
                    if (train)
                        best_child_index = findBestChildVisitCountStochastic(MCTSRootNodeNN2);
                    else
                        best_child_index = findBestChildVisitCount(MCTSRootNodeNN2);

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
        public int RecordOneGameGPU(List<Tuple<int, int>> history, List<List<float>> policies, Player aEvaluationNetworkPlayer,
            NeuralNetwork NN1, Node<TicTacToePosition> rootNode,  bool train)
        {
            TicTacToeGame game = new TicTacToeGame();
            Node<TicTacToePosition> MCTSRootNodeNN1 = rootNode;

            Queue<Node<TicTacToePosition>> pendingNN1Requests = new Queue<Node<TicTacToePosition>>();

            for (int curr_ply = 0; curr_ply < Params.MAXIMUM_PLYS; ++curr_ply)  // we always finish the game for tic tac toe
            {
                MCTSRootNodeNN1.Value = new TicTacToePosition(game.position);

                if (game.IsOver())
                {
                    break;
                }

                createChildren(MCTSRootNodeNN1);

                /* find value, policy */
                if (MCTSRootNodeNN1.nn_policy == null)
                {
                    calculateNNOutputGPU(MCTSRootNodeNN1, NN1, pendingNN1Requests);
                }

                int nofSimsPerMove = train ? Params.NOF_SIMS_PER_MOVE_TRAINING : Params.NOF_SIMS_PER_MOVE_TESTING;

                for (int simulation = 0; simulation < nofSimsPerMove; ++simulation)
                {
                    Tuple<float[], float> result = NN1.GetResultAsync(); // try to get a result, if there is one, try to get more
                    while (result != null)
                    {
                        Node<TicTacToePosition> nodeToUpdate = pendingNN1Requests.Dequeue();
                        normalizePolicy(nodeToUpdate, result.Item1, curr_ply);
                        nodeToUpdate.nn_value = result.Item2;
                        nodeToUpdate.waitingForGPUPrediction = false;
                        removeVirtualLoss(MCTSRootNodeNN1, nodeToUpdate);
                        backpropagateScore(MCTSRootNodeNN1, nodeToUpdate, nodeToUpdate.nn_value);

                        result = NN1.GetResultAsync(); // try to get a result, if there is one, try to get more
                    }
                    while (pendingNN1Requests.Count > Params.MAX_PENDING_NN_EVALS)
                    {
                        // if we need to wait then wait
                        result = NN1.GetResultSync();
                        Node<TicTacToePosition> nodeToUpdate = pendingNN1Requests.Dequeue();
                        normalizePolicy(nodeToUpdate, result.Item1, curr_ply);
                        nodeToUpdate.nn_value = result.Item2;
                        nodeToUpdate.waitingForGPUPrediction = false;
                        removeVirtualLoss(MCTSRootNodeNN1, nodeToUpdate);
                        backpropagateScore(MCTSRootNodeNN1, nodeToUpdate, nodeToUpdate.nn_value);
                    }
                    SearchUsingNNGPU(MCTSRootNodeNN1, NN1, pendingNN1Requests); // expand tree and improve accuracy at MCTSRootNode
                }
                // wait for all search results before deciding on which move to play ( because of virtual losses)
                while (pendingNN1Requests.Count > 0)
                {
                    // if we need to wait then wait
                    Tuple<float[], float> result = NN1.GetResultSync();
                    Node<TicTacToePosition> nodeToUpdate = pendingNN1Requests.Dequeue();
                    normalizePolicy(nodeToUpdate, result.Item1, curr_ply);
                    nodeToUpdate.nn_value = result.Item2;
                    nodeToUpdate.waitingForGPUPrediction = false;
                    removeVirtualLoss(MCTSRootNodeNN1, nodeToUpdate);
                    backpropagateScore(MCTSRootNodeNN1, nodeToUpdate, nodeToUpdate.nn_value);
                }

                // after search, record root node visits for new policy vector

                List<float> policy = Enumerable.Repeat(0.0f, 25).ToList();
                float totalVisits = 0;
                for (int i = 0; i < MCTSRootNodeNN1.Children.Count; ++i)
                {
                    policy[MCTSRootNodeNN1.Children[i].moveIndex] = MCTSRootNodeNN1.Children[i].visits;
                    totalVisits += MCTSRootNodeNN1.Children[i].visits;
                }
                for (int i = 0; i < MCTSRootNodeNN1.Children.Count; ++i)
                {
                    policy[MCTSRootNodeNN1.Children[i].moveIndex] /= totalVisits;
                    if (policy[MCTSRootNodeNN1.Children[i].moveIndex] == float.NaN ||
                        policy[MCTSRootNodeNN1.Children[i].moveIndex] == float.NegativeInfinity ||
                        policy[MCTSRootNodeNN1.Children[i].moveIndex] == float.PositiveInfinity)
                    {
                        policy[MCTSRootNodeNN1.Children[i].moveIndex] = 0.0f;
                    }
                }
                policies.Add(policy);
                
                int best_child_index = -1;
                if (train && curr_ply < Params.STOCHASTIC_MOVES_FIRST_X_MOVES)
                    best_child_index = findBestChildVisitCountStochastic(MCTSRootNodeNN1);
                else
                    best_child_index = findBestChildVisitCount(MCTSRootNodeNN1);

                List<Tuple<int, int>> moves = game.GetMoves();
                Tuple<int, int> move = moves[best_child_index];
                game.DoMove(move);
                history.Add(move);

                /* tree re-use */
                MCTSRootNodeNN1 = MCTSRootNodeNN1.Children[best_child_index];
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
                    break;
                }
                createChildren(MCTSRootNodeNN1);
                createChildren(MCTSRootNodeNN2);

                if (MCTSRootNodeNN1.nn_policy == null)
                {
                    calculateNNOutputGPU(MCTSRootNodeNN1, NN1, pendingNN1Requests);
                }
                if (MCTSRootNodeNN2.nn_policy == null)
                {
                    calculateNNOutputGPU(MCTSRootNodeNN2, NN2, pendingNN2Requests);
                }
                int maxSimulations = train ? Params.NOF_SIMS_PER_MOVE_TRAINING : Params.NOF_SIMS_PER_MOVE_TESTING;
                
                for (int simulation = 0; simulation < maxSimulations; ++simulation)
                {
                    if (curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                    {
                        Tuple<float[], float> result = NN1.GetResultAsync(); // try to get a result, if there is one, try to get more
                        while (result != null)
                        {
                            Node<TicTacToePosition> nodeToUpdate = pendingNN1Requests.Dequeue();
                            normalizePolicy(nodeToUpdate, result.Item1, curr_ply);
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
                            normalizePolicy(nodeToUpdate, result.Item1, curr_ply);
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
                            normalizePolicy(nodeToUpdate, result.Item1, curr_ply);
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
                            normalizePolicy(nodeToUpdate, result.Item1, curr_ply);
                            nodeToUpdate.nn_value = result.Item2;
                            nodeToUpdate.waitingForGPUPrediction = false;
                            removeVirtualLoss(nodeToUpdate);
                            backpropagateScore(nodeToUpdate, nodeToUpdate.nn_value);
                        }
                        SearchUsingNNGPU(MCTSRootNodeNN2, NN2, pendingNN2Requests); // expand tree and improve accuracy at MCTSRootNode
                    }
                }
                // wait for all search results before deciding on which move to play
                while (pendingNN1Requests.Count > 0)
                {
                    // if we need to wait then wait
                    Tuple<float[], float> result = NN1.GetResultSync();
                    Node<TicTacToePosition> nodeToUpdate = pendingNN1Requests.Dequeue();
                    normalizePolicy(nodeToUpdate, result.Item1, curr_ply);
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
                    normalizePolicy(nodeToUpdate, result.Item1, curr_ply);
                    nodeToUpdate.nn_value = result.Item2;
                    nodeToUpdate.waitingForGPUPrediction = false;
                    removeVirtualLoss(nodeToUpdate);
                    backpropagateScore(nodeToUpdate, nodeToUpdate.nn_value);
                }

                int best_child_index = -1;
                if (curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                {
                    if (train &&curr_ply<10 ||curr_ply<6)
                        best_child_index = findBestChildVisitCountStochastic(MCTSRootNodeNN1);
                    else
                        best_child_index = findBestChildVisitCount(MCTSRootNodeNN1);
                }
                else
                {
                    if (train&&curr_ply<10 || curr_ply<6)
                        best_child_index = findBestChildVisitCountStochastic(MCTSRootNodeNN2);
                    else
                        best_child_index = findBestChildVisitCount(MCTSRootNodeNN2);
                }

                List<Tuple<int, int>> moves = game.GetMoves();
                Tuple<int, int> move = moves[best_child_index];
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
            game = new TicTacToeGame(currNode.Value);
            if (game.IsOver())
            {
                /* update the tree with the new score and visit counts */
                backpropagateScore(currNode, game.GetScore());
            }
            else
            {
                /* create children if possible */
                createChildren(currNode);

                if (currNode.nn_policy == null)
                {
                    calculateNNOutput(currNode, NN, false);
                }

                /* update the tree with the new score and visit counts */
                backpropagateScore(currNode, currNode.nn_value); 
            }
        }

        /// <summary>
        /// The Search uses a tree Nodes<TicTacToePosition> and expands it until it runs out of time
        /// Nodes which look promising according to the NN are expanded greedily
        /// </summary>
        /// <param name="currNode"></param>
        /// <returns>Eval</returns>
        private void SearchUsingNNGPU(Node<TicTacToePosition> currRootNode, NeuralNetwork NN, Queue<Node<TicTacToePosition>> queue)
        {
            TicTacToeGame game = new TicTacToeGame(currRootNode.Value);
            List<Tuple<int, int>> moves = game.GetMoves();

            /* find the most promising leaf node */
            Node<TicTacToePosition> currNode = findMostPromisingLeafNode(currRootNode);

            /* if the leaf node is a game ending state use correct score */
            game = new TicTacToeGame(currNode.Value);
            if (game.IsOver())
            {
                backpropagateScore(currRootNode, currNode, game.GetScore());
            }
            else
            {
                /* create children if possible */
                createChildren(currNode);

                if (currNode.nn_policy == null && !currNode.waitingForGPUPrediction)
                {
                    calculateNNOutputGPU(currNode, NN, queue);
                }
            }
        }
        /// <summary>
        /// Loss for both players
        /// </summary>
        /// <param name="currNode"></param>
        private void propagateVirtualLoss(Node<TicTacToePosition> currRootnode, Node<TicTacToePosition> currNode)
        {
            while (currNode != currRootnode)
            {
                currNode.virtualVisits += 1;
                currNode.score_sum -= 1.0f;
                currNode.q_value = currNode.score_sum / (currNode.visits + currNode.virtualVisits);
                currNode = currNode.GetParent();
            }
            currRootnode.virtualVisits += 1;
            currRootnode.score_sum -= 1.0f;
            currRootnode.q_value = currRootnode.score_sum / (currRootnode.visits + currRootnode.virtualVisits);
        }
        private void propagateVirtualLoss(Node<TicTacToePosition> currNode)
        {
            while (currNode != null)
            {
                currNode.virtualVisits += 1;
                currNode.score_sum -= 1.0f;
                currNode.q_value = currNode.score_sum / (currNode.visits + currNode.virtualVisits);
                currNode = currNode.GetParent();
            }
        }
        private void removeVirtualLoss(Node<TicTacToePosition> currNode)
        {
            // we store the q_value for the opposite player in the node, during search we look at the next level
            while (currNode != null)
            {
                currNode.virtualVisits -= 1;
                currNode.score_sum += 1.0f;
                currNode.q_value = currNode.score_sum / (currNode.visits + currNode.virtualVisits);
                currNode = currNode.GetParent();
            }
        }
        private void removeVirtualLoss(Node<TicTacToePosition> currRootnode, Node<TicTacToePosition> currNode)
        {
            while (currNode != currRootnode)
            {
                currNode.virtualVisits -= 1;
                currNode.score_sum += 1.0f;
                currNode.q_value = currNode.score_sum / (currNode.visits + currNode.virtualVisits);
                currNode = currNode.GetParent();
            }
            currRootnode.virtualVisits -= 1;
            currRootnode.score_sum += 1.0f;
            currRootnode.q_value = currNode.score_sum / (currRootnode.visits + currRootnode.virtualVisits);
        }
        private void printNNOutput(NeuralNetwork nn)
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

            game = new TicTacToeGame();

            game.DoMove(Tuple.Create(0, 0));
            game.DoMove(Tuple.Create(0, 1));
            game.DoMove(Tuple.Create(0, 2));

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
        private float getNoiseWeight()
        {
            return 0.0f;
        }
        private int findBestChildVisitCountStochastic(Node<TicTacToePosition> currNode)
        {
            float randomNr = RandomGen2.NextFloat();

            float probabilitySum = 0.0f;
            float sumVisits = 0.0f;
            List<float> moveProbabilities = new List<float>(new float[currNode.Children.Count]);

            for (int i = 0; i < currNode.Children.Count; ++i)
            {
                sumVisits += currNode.Children[i].visits;
            }
            for (int i = 0; i < currNode.Children.Count; ++i)
            {
                moveProbabilities[i] = currNode.Children[i].visits/sumVisits;
            }
            for (int i = 0; i < currNode.Children.Count; ++i)
            {
                probabilitySum += moveProbabilities[i];
                if(probabilitySum >= randomNr)
                {
                    return i;
                }
            }
            return currNode.Children.Count-1;
        }
        private int findBestChildVisitCount(Node<TicTacToePosition> currNode)
        {
            float best_visit_count = -1;
            int best_child_index = -1;

            for (int i = 0; i < currNode.Children.Count; ++i)
            {
                float tempvisitCount = currNode.Children[i].visits;
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
        private void normalizePolicy(Node<TicTacToePosition> currNode, float[] rawPolicy, int depth)
        {
            currNode.nn_policy = new List<float>(new float[rawPolicy.Length]);

            /* re-normalize policy vector */
            float sum = 0;
            for (int i = 0; i < currNode.Children.Count; ++i)
            {
                sum += rawPolicy[currNode.Children[i].moveIndex];
            }

            if (sum > 0)
            {
                for (int i = 0; i < currNode.Children.Count; ++i)
                {
                    currNode.nn_policy[currNode.Children[i].moveIndex] = rawPolicy[currNode.Children[i].moveIndex] / sum;
                }
            }
            if (currNode.Children.Count > 0)
            {
                DirichletNoise dn = new DirichletNoise(currNode.Children.Count);
                for (int i = 0; i < currNode.Children.Count; ++i)
                {
                    float noise = dn.GetNoise(i);
                    currNode.nn_policy[currNode.Children[i].moveIndex] =
                        currNode.nn_policy[currNode.Children[i].moveIndex] * (1 - getNoiseWeight()) + getNoiseWeight() * noise;
                }
            }
        }
        private void calculateNNOutput(Node<TicTacToePosition> currNode, NeuralNetwork NN, bool rootNode)
        {
            Tuple<float[], float> prediction = NN.Predict(currNode.Value);

            currNode.nn_value = prediction.Item2;
            currNode.nn_policy = new List<float>(new float[prediction.Item1.Length]);

            /* re-normalize policy vector */
            float sum = 0;
            for (int i = 0; i < currNode.Children.Count; ++i)
            {
                sum += prediction.Item1[currNode.Children[i].moveIndex];
            }

            if (sum > 0)
            {
                for (int i = 0; i < currNode.Children.Count; ++i)
                {
                    currNode.nn_policy[currNode.Children[i].moveIndex] = prediction.Item1[currNode.Children[i].moveIndex] / sum;
                }
            }
            if (currNode.Children.Count > 0)
            {
                DirichletNoise dn = new DirichletNoise(currNode.Children.Count);
                for (int i = 0; i < currNode.Children.Count; ++i)
                {
                    float noise = dn.GetNoise(i);
                    currNode.nn_policy[currNode.Children[i].moveIndex] =
                        currNode.nn_policy[currNode.Children[i].moveIndex] * (1 - getNoiseWeight()) + getNoiseWeight() * noise;
                }
            }
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
        private void calculateNNOutputGPU(Node<TicTacToePosition> currRootnode, Node<TicTacToePosition> currNode, NeuralNetwork NN, Queue<Node<TicTacToePosition>> queue)
        {
            currNode.waitingForGPUPrediction = true;
            propagateVirtualLoss(currRootnode, currNode);
            NN.PredictGPU(currNode.Value);
            queue.Enqueue(currNode);
        }
        private Node<TicTacToePosition> findMostPromisingLeafNode(Node<TicTacToePosition> currNode)
        {
            while (currNode.HasChild)
            {
                List<int> draws = new List<int>();
                /* create the game from the TicTacToePosition */
                TicTacToeGame game = new TicTacToeGame(currNode.Value);
                List<Tuple<int, int>> moves = game.GetMoves(); // valid moves

                /* find best child node (best UCT value) to expand */
                float bestUCTScore = float.NegativeInfinity;
                int bestChildIndex = -1;

                // if nnpolicy is null then also all children have no nn output, but possibly a score from endgame position
                for (int i = 0; i < currNode.Children.Count; ++i)
                {
                    float temp_UCT_score = float.NegativeInfinity;

                    // q_value
                    float childWinrate = currNode.Children[i].q_value;

                    // exploration
                    float explorationTerm = 0.0f;
                    if (currNode.nn_policy != null)
                    {
                        // we have the policy output
                        explorationTerm = Params.C_PUCT * currNode.nn_policy[currNode.Children[i].moveIndex] *
                            (float)Math.Sqrt(currNode.visits + currNode.virtualVisits) / (float)(currNode.Children[i].visits +
                            currNode.Children[i].virtualVisits + 1);
                    }
                    else
                    {
                        // assume policy equal for all children if not found yet (because of virtual visits)
                        explorationTerm = Params.C_PUCT * (1.0f / currNode.Children.Count) *
                            (float)Math.Sqrt(currNode.visits + currNode.virtualVisits) / (float)(currNode.Children[i].visits +
                            +currNode.Children[i].virtualVisits + 1);
                    }

                    // bonus for unvisited
                    if (currNode.Children[i].visits == 0 && currNode.Children[i].virtualVisits == 0)
                    {
                        temp_UCT_score = childWinrate + explorationTerm + Params.FPU_VALUE;
                    }
                    else
                    {
                        temp_UCT_score = childWinrate + explorationTerm;
                    }

                    if (temp_UCT_score > bestUCTScore)
                    {
                        draws.Clear();
                        bestChildIndex = i;
                        bestUCTScore = temp_UCT_score;
                    }
                    else if (temp_UCT_score == bestUCTScore)
                    {
                        draws.Add(i);
                    }
                    //Console.WriteLine("winrate " + childWinrate + " exploration " + explorationTerm + " total " + temp_UCT_score);
                }
                if (draws.Count != 0)
                {
                    currNode = currNode.Children[draws[RandomGen2.Next(0,draws.Count)]];
                    //currNode = currNode.Children[draws[0]];
                    //Console.WriteLine("There was a draw on choosing most promising child node");
                }
                else
                {
                    currNode = currNode.Children[bestChildIndex];
                }
            }
            return currNode;
        }
        private void backpropagateScore(Node<TicTacToePosition> currNode, float score)
        {
            // argument score is from the view of player X always
            // we store the q_value for the opposite player in the node, during search we look at the next level
            if (currNode.Value.sideToMove == Player.X)
                score *= -1;

            while (currNode != null)
            {
                currNode.score_sum += score;
                currNode.visits += 1;
                currNode.q_value = currNode.score_sum / (currNode.visits + currNode.virtualVisits);
                currNode = currNode.GetParent();
                score *= -1;
            }
        }
        private void backpropagateScore(Node<TicTacToePosition> currRootnode, Node<TicTacToePosition> currNode, float score)
        {
            // argument score is from the view of player X always
            // we store the q_value for the opposite player in the node, during search we look at the next level
            if (currNode.Value.sideToMove == Player.X)
                score *= -1;

            while (currNode != currRootnode)
            {
                currNode.score_sum += score;
                currNode.visits += 1;
                currNode.q_value = currNode.score_sum / (currNode.visits + currNode.virtualVisits);
                currNode = currNode.GetParent();
                score *= -1;
            }
            currRootnode.score_sum += score;
            currRootnode.visits += 1;
            currRootnode.q_value = currRootnode.score_sum / (currRootnode.visits + currRootnode.virtualVisits);
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
                score = game.GetScore();
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

                score = simulateRandomPlayout(currNode);
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
            if (node.visits != 0)
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
