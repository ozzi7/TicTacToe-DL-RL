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
        private float winLossDiff = 0.0f;
        private float trainingXwins = 0;
        private float trainingZwins = 0;
        private float trainingMoves = 0;
        private float startBoardValue = 0;

        public Trainer(NeuralNetwork aCurrentNN)
        {
            currentNN = aCurrentNN;
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
            bool continuewithpreviousrun = true;

            bestNN = new NeuralNetwork(currentNN.weights);
            printNNOutputs(bestNN);

            if (!continuewithpreviousrun)
            {
                Params.DIRICHLET_NOISE_WEIGHT = 0.0f;
                Params.USE_REAL_TERMINAL_VALUES = true;
                Params.GPU_ENABLED = false;

                CheckPerformanceVsRandomKeras(bestNN, Params.NOF_GAMES_VS_RANDOM);

                WritePlotStatistics();
            }
            for (int i = 0; i < Params.NOF_EPOCHS; ++i)
            {
                if (!continuewithpreviousrun)
                {
                    Console.WriteLine("Main Thread: Epoch start");

                    //// #################################### CREATE NEW TRAINING GAMES ##########################################

                    Params.DIRICHLET_NOISE_WEIGHT = 0.2f; 
                    Params.USE_REAL_TERMINAL_VALUES = true;
                    Params.GPU_ENABLED = true;
                    ProduceTrainingGamesKeras(bestNN, Params.NOF_GAMES_TRAIN_KERAS);

                    // ##################################### TRAIN NETWORK WEIGHTS ############################################

                }
                ProcessStartInfo pythonInfo = new ProcessStartInfo();
                Process python;
                pythonInfo.FileName = @"python.exe";
                pythonInfo.Arguments = "\"Z:\\CloudStation\\GitHub Projects\\TicTacToe-DL-RL\\Training\\main.py \" doesntmatter"; // TODO: should be relative
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
                Params.USE_REAL_TERMINAL_VALUES = false; //
                bool newBestFound = CheckPerformanceVsOldNet(currentNN, bestNN, Params.NOF_GAMES_TEST);

                if (!newBestFound)
                {
                    printNNOutputs(bestNN);
                }
                else
                { 
                    Console.WriteLine("New best network found!");
                    printNNOutputs(currentNN);

                    bestNN = new NeuralNetwork(currentNN.weights);

                    Params.DIRICHLET_NOISE_WEIGHT = 0.0f;
                    Params.USE_REAL_TERMINAL_VALUES = true;
                    Params.GPU_ENABLED = false;
                    CheckPerformanceVsRandomKeras(bestNN, Params.NOF_GAMES_VS_RANDOM);
                }

                WritePlotStatistics();
                continuewithpreviousrun = false;
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
                nn2.SetIds(i,i); // TODO: 
                nn2.ChannelInit();
                nn2.EnqueueWeights();
            }
            OpenCL.CreateNetworkWeightBuffers();
            Thread thread = new Thread(OpenCL.Run);
            thread.Start();

            for (int z = 0; z < 10; ++z)
            {
                Game game = new Game();
                game.DoMove(Tuple.Create(3, 2));
                game.DoMove(Tuple.Create(0, 1));
                game.DoMove(Tuple.Create(0, 4)); // y, x
                game.DoMove(Tuple.Create(2, 4));

                nn1.PredictGPU(game.position);
                nn2.PredictGPU(game.position);


                // if we need to wait then wait
                Tuple<float[], float> prediction1 = nn1.GetResultSync();
                Tuple<float[], float> prediction2 = nn2.GetResultSync();

                for (int i = 0; i < GameProperties.GAMEBOARD_WIDTH; ++i)
                {
                    Console.WriteLine(prediction1.Item1[i * GameProperties.GAMEBOARD_WIDTH + 0].ToString("0.000") + " " +
                    prediction1.Item1[i * GameProperties.GAMEBOARD_WIDTH + 1].ToString("0.000") + " " +
                    prediction1.Item1[i * GameProperties.GAMEBOARD_WIDTH + 2].ToString("0.000") + " " +
                    prediction1.Item1[i * GameProperties.GAMEBOARD_WIDTH + 3].ToString("0.000") + " " +
                    prediction1.Item1[i * GameProperties.GAMEBOARD_WIDTH + 4].ToString("0.000"));
                }
                Console.WriteLine("Value " + prediction1.Item2);
                Console.WriteLine("\n");

                for (int i = 0; i < GameProperties.GAMEBOARD_WIDTH; ++i)
                {
                    Console.WriteLine(prediction2.Item1[i * GameProperties.GAMEBOARD_WIDTH + 0].ToString("0.000") + " " +
                    prediction2.Item1[i * GameProperties.GAMEBOARD_WIDTH + 1].ToString("0.000") + " " +
                    prediction2.Item1[i * GameProperties.GAMEBOARD_WIDTH + 2].ToString("0.000") + " " +
                    prediction2.Item1[i * GameProperties.GAMEBOARD_WIDTH + 3].ToString("0.000") + " " +
                    prediction2.Item1[i * GameProperties.GAMEBOARD_WIDTH + 4].ToString("0.000"));
                }
                Console.WriteLine("Value " + prediction2.Item2);
                Console.WriteLine("\n");
            }
            thread.Abort();
            ValidateOutputCPU();
        }
        public void ValidateOutputCPU()
        {
            Game game = new Game();
            game.DoMove(Tuple.Create(3, 2));
            game.DoMove(Tuple.Create(0, 1));
            game.DoMove(Tuple.Create(0, 4)); // y, x
            game.DoMove(Tuple.Create(2, 4));

            Tuple<float[], float> prediction = currentNN.Predict(game.position);

            for (int i = 0; i < GameProperties.GAMEBOARD_WIDTH; ++i)
            {
                Console.WriteLine(prediction.Item1[i * GameProperties.GAMEBOARD_WIDTH + 0].ToString("0.000") + " " +
                prediction.Item1[i * GameProperties.GAMEBOARD_WIDTH + 1].ToString("0.000") + " " +
                prediction.Item1[i * GameProperties.GAMEBOARD_WIDTH + 2].ToString("0.000") + " " +
                prediction.Item1[i * GameProperties.GAMEBOARD_WIDTH + 3].ToString("0.000") + " " +
                prediction.Item1[i * GameProperties.GAMEBOARD_WIDTH + 4].ToString("0.000"));
            }
            Console.WriteLine("Value " + prediction.Item2);
            Console.WriteLine("\n");
        }
        /// <summary>
        /// Keras only, returns whether network should be replaced
        /// </summary>
        /// <param name="nofGames"></param>
        /// <param name="NNtoEval"></param>
        /// <param name="oldNN"></param>
        public bool CheckPerformanceVsOldNet(NeuralNetwork NNtoEval, NeuralNetwork oldNN, int nofGames)
        {
            Stopwatch sw = new Stopwatch();
            sw.Start();

            List<int> wins = new List<int>(new int[nofGames]);
            List<int> draws = new List<int>(new int[nofGames]);
            List<int> losses = new List<int>(new int[nofGames]);
            List<int> movecount = new List<int>(new int[nofGames]);
            List<int> winsX = new List<int>(new int[nofGames]);
            List<int> winsZ = new List<int>(new int[nofGames]);
            List<List<Tuple<int, int>>> histories = new List<List<Tuple<int, int>>>();

            List<NeuralNetwork> prevNns = new List<NeuralNetwork>();
            List<NeuralNetwork> evalNNs = new List<NeuralNetwork>();

            for (int i = 0; i < nofGames; ++i)
            {
                histories.Add(new List<Tuple<int, int>>());
            }

            // ################################# COPY WEIGHTS TO GPU MEMORY ###########################################
            ID.ResetGlobalID();
            OpenCL.ClearWeights();

            NeuralNetwork evalNN = new NeuralNetwork(NNtoEval.weights);
            NeuralNetwork prevNN = new NeuralNetwork(oldNN.weights);
            evalNN.SetIds(0, 0);
            prevNN.SetIds(1, 1);

            evalNN.EnqueueWeights();
            prevNN.EnqueueWeights();
            evalNN.ChannelInit();
            prevNN.ChannelInit();

            evalNNs.Add(evalNN);
            prevNns.Add(prevNN);

            for (int i = 1; i < nofGames; ++i)
            {
                evalNN = new NeuralNetwork();
                evalNN.DeleteArrays();
                evalNN.SetIds(0, i*2);
                evalNN.ChannelInit();

                prevNN = new NeuralNetwork();
                prevNN.DeleteArrays();
                prevNN.SetIds(1, i*2+1);
                prevNN.ChannelInit();

                evalNNs.Add(evalNN);
                prevNns.Add(prevNN);
            }

            OpenCL.CreateNetworkWeightBuffers();

            // ###################################### GPU LOOP ##############################################


            Console.WriteLine("Main Thread: GPU test games starting...");
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
                        Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;

                        int result = PlayOneGameGPU(histories[i], evaluationNetworkPlayer, 
                            evalNNs[i], prevNns[i], false);

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

                        movecount[i] += histories[i].Count;

                        Interlocked.Add(ref sharedLoopCounter, 1);
                        progress.Report((double)Interlocked.Read(ref sharedLoopCounter) / nofGames);
                    });
            }
            thread.Abort();
      
            if(true)
            {
                /* to display games */
                for(int i = 0; i < nofGames; ++i)
                {
                    if ((i % 2) == 0)
                        Console.WriteLine("Eval player playing as Player X");
                    else
                        Console.WriteLine("Eval player playing as Player Z");
                    Game game = new Game();
                    game.DisplayHistory(histories[i]);
                }
            }
            bool foundNewBest = false;
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
                foundNewBest = true;

                currentPseudoELO += (float)(totalWins - (nofGames-(totalDraws+totalWins))) / (float)nofGames;
            }
            winLossDiff += (float)(totalWins - (nofGames - (totalDraws + totalWins))) / (float)nofGames;

            sw.Stop();
            Console.WriteLine("Main Thread: Vs. previous best: W/D/L : " + totalWins + " " + totalDraws + " " + (nofGames - (totalDraws + totalWins)) + " - " +
                Math.Round((((totalWins + totalDraws * 0.5f) / nofGames) * 100.0f), 2) + "%");
            Console.WriteLine("Main Thread: Finished in: " + sw.ElapsedMilliseconds + "ms");

            return foundNewBest;
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
                    + " " + Math.Round(winrateVsRandMovingAvg2.Average, 2) + " " + Math.Round(winrateVsRandMovingAvg3.Average, 2)
                    + " " + Math.Round(winLossDiff, 2)
                    + " " + Math.Round(trainingXwins, 2)
                    + " " + Math.Round(trainingZwins, 2)
                    + " " + Math.Round(trainingMoves, 2)
                    + " " + Math.Round(startBoardValue, 3));
            }
        }
        /// <summary>
        /// Return file name for games
        /// </summary>
        /// <param name="nofGames"></param>
        public void ProduceTrainingGamesKeras(NeuralNetwork nn, int nofGames)
        {
            Console.WriteLine("Main Thread: Creating " + nofGames + " training games...");

            List<NeuralNetwork> nns = new List<NeuralNetwork>();

            // ################################# COPY WEIGHTS TO GPU MEMORY ###########################################

            ID.ResetGlobalID();
            OpenCL.ClearWeights();

            for (int i = 0; i < Params.NOF_CPU_THREADS_GPU_WORKLOAD; ++i)
            {
                NeuralNetwork playingNNlocal = new NeuralNetwork();
                playingNNlocal.DeleteArrays();
                playingNNlocal.SetIds(0, ID.GetGlobalID());
                playingNNlocal.ChannelInit();

                nns.Add(playingNNlocal);
            }

            nns[0] = new NeuralNetwork(nn.weights);
            nns[0].SetIds(0, 0);
            nns[0].ChannelInit();
            nns[0].EnqueueWeights();
            OpenCL.CreateNetworkWeightBuffers();
            

            List<List<Tuple<int, int>>> moves = new List<List<Tuple<int, int>>>(nofGames);
            List<List<List<float>>> policies = new List<List<List<float>>>(nofGames);

            for (int i = 0; i < nofGames; ++i)
            {
                moves.Add(new List<Tuple<int, int>>());
                policies.Add(new List<List<float>>());
            }
            List<float> scores = new List<float>(nofGames);
            scores.AddRange(Enumerable.Repeat(0.0f, nofGames));

            // ###################################### GPU LOOP ##############################################

            trainingXwins = 0;
            trainingZwins = 0;
            trainingMoves = 0;

            Thread thread = new Thread(OpenCL.Run);
            thread.Priority = ThreadPriority.Highest;
            thread.Start();

            using (var progress = new ProgressBar())
            {
                long sharedLoopCounter = 0;

                ThreadPool.SetMinThreads(Params.NOF_CPU_THREADS_GPU_WORKLOAD, Params.NOF_CPU_THREADS_GPU_WORKLOAD);
                for (int j = 0; j < nofGames / Params.NOF_CPU_THREADS_GPU_WORKLOAD; ++j)
                { // process batches of games to re-use neural networks
                    Parallel.For(j * Params.NOF_CPU_THREADS_GPU_WORKLOAD, j * Params.NOF_CPU_THREADS_GPU_WORKLOAD + Params.NOF_CPU_THREADS_GPU_WORKLOAD,
                        new ParallelOptions { MaxDegreeOfParallelism = Params.NOF_CPU_THREADS_GPU_WORKLOAD }, i =>
                    {
                        Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z; // doesnt really matter for 2 equal networks
                        scores[i] = RecordOneGameGPU(moves[i], policies[i], evaluationNetworkPlayer,
                            nns[i % Params.NOF_CPU_THREADS_GPU_WORKLOAD]);
                        Interlocked.Add(ref sharedLoopCounter, 1);
                        progress.Report((double)Interlocked.Read(ref sharedLoopCounter) / nofGames);
                    });
                }
            }
            thread.Abort();

            for (int i = 0; i < scores.Count; ++i)
            {
                if (scores[i] == 1.0f)
                    trainingXwins++;
                else if (scores[i] == -1.0f)
                    trainingZwins++;
                trainingMoves += moves[i].Count;
            }
            trainingMoves /= nofGames;
            trainingZwins /= nofGames;
            trainingXwins /= nofGames;
            

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

            // ###################################### GPU TRAINING LOOP ##############################################
            sw.Stop();
            Console.WriteLine("Main Thread: Finished in: " + sw.ElapsedMilliseconds +"ms");
            sw.Reset();
            sw.Start();

            Params.DIRICHLET_NOISE_WEIGHT = 0.1f;
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

                WaitHandle.WaitAll(waitHandles);
                thread.Abort();
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
           
            sw.Stop();
            Console.WriteLine("Main Thread: Finished in: " + sw.ElapsedMilliseconds + "ms");
            sw.Reset();
            sw.Start();

            // #################################### GPU TEST LOOP ##########################################


            Console.WriteLine("Main Thread: GPU test games starting...");

            Params.DIRICHLET_NOISE_WEIGHT = 0.1f;
            thread = new Thread(OpenCL.Run);
            thread.Priority = ThreadPriority.Highest;
            thread.Start();

            numOfThreads = Math.Min(Params.NOF_GAMES_TEST, Params.NOF_CPU_THREADS_GPU_WORKLOAD);
            waitHandles = new WaitHandle[numOfThreads];

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
                        Game game = new Game();
                        game.DisplayHistory(history);
                    }
                    handle.Set();
                });
                waitHandles[jk] = handle;
                thread2.Start();
            }
            WaitHandle.WaitAll(waitHandles);
            thread.Abort();
           
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

                printNNOutputs(bestNN);
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
            NNPlayer nnPlayer = new NNPlayer(NN);
            RandomPlayer randPlayer = new RandomPlayer();
            Game game = new Game();

            float result = 0.0f;

            for (int curr_ply = 0; curr_ply < GameProperties.MAXIMUM_PLYS; ++curr_ply)
            {
                if (game.IsOver())
                {
                    result = game.GetScore();

                    if (evaluationNetworkPlayer == Player.X && result == 1 ||
                        evaluationNetworkPlayer == Player.Z && result == -1)
                    {
                        return 1;
                    }
                    else
                        return 0;
                }

                Tuple<int, int> move;
                if (game.position.sideToMove == evaluationNetworkPlayer)
                {
                    move = nnPlayer.GetMove(game, nofSimsPerMove);
                }
                else
                {
                    move = randPlayer.GetMove(game);
                }

                game.DoMove(move);
                nnPlayer.DoMove(move);
                history.Add(move);
            }
            return 0;
        }
        /// <summary>
        /// 
        /// </summary>
        /// <param name="history"></param>
        /// <param name="aEvaluationNetworkPlayer"></param>
        /// <param name="NN1"></param>
        /// <param name="nn2"></param>
        /// <returns>Return 0 for draw, win for X 1, win for Z -1 </returns>
        public int PlayOneGame(List<Tuple<int, int>> history, Player aEvaluationNetworkPlayer, NeuralNetwork nn1, NeuralNetwork nn2, bool train)
        {
            Game game = new Game();
            NNPlayer nn1Player = new NNPlayer(nn1);
            NNPlayer nn2Player = new NNPlayer(nn2);

            for (int curr_ply = 0; curr_ply < GameProperties.MAXIMUM_PLYS; ++curr_ply)  // we always finish the game for tic tac toe
            {
                if (game.IsOver()) {
                    return game.GetScore();
                }
                Tuple<int, int> move;
                int nofSims = train ? Params.NOF_SIMS_PER_MOVE_TRAINING : Params.NOF_SIMS_PER_MOVE_TRAINING;
                if (game.position.sideToMove == aEvaluationNetworkPlayer)
                {
                    if(train)
                        move = nn1Player.GetMove(game, nofSims);
                    else
                        move = nn1Player.GetMoveStochastic(game, nofSims);
                }
                else
                {
                    if (train)
                        move = nn2Player.GetMove(game, nofSims);
                    else
                        move = nn2Player.GetMoveStochastic(game, nofSims);
                }

                game.DoMove(move);
                nn1Player.DoMove(move);
                nn2Player.DoMove(move);
                history.Add(move);
            }

            return game.GetScore();
        }
        /// <summary>
        /// Used for training games
        /// </summary>
        /// <param name="history"></param>
        /// <param name="aEvaluationNetworkPlayer"></param>
        /// <param name="NN1"></param>
        /// <param name="NN2"></param>
        /// <returns>Return 0 for draw, win for X 1, win for Z -1 </returns>
        public int RecordOneGameGPU(List<Tuple<int, int>> history, List<List<float>> policies, Player aEvaluationNetworkPlayer,
            NeuralNetwork nn)
        {
            Game game = new Game();
            NNPlayer nn1Player = new NNPlayer(nn);
            NNPlayer nn2Player = new NNPlayer(nn);

            for (int curr_ply = 0; curr_ply < GameProperties.MAXIMUM_PLYS; ++curr_ply)
            {
                if (game.IsOver())
                {
                    break;
                }
                Tuple<int, int> move = Tuple.Create<int,int>(-1,-1);

                if (game.position.sideToMove == aEvaluationNetworkPlayer)
                {
                    if (curr_ply < Params.STOCHASTIC_MOVES_FIRST_X_MOVES_TRAINING)
                        move = nn1Player.GetMoveStochastic(game, Params.NOF_SIMS_PER_MOVE_TRAINING);
                    else
                        move = nn1Player.GetMove(game, Params.NOF_SIMS_PER_MOVE_TRAINING);
                    policies.Add(nn1Player.mcts.GetPolicy());
                }
                else
                {
                    if (curr_ply < Params.STOCHASTIC_MOVES_FIRST_X_MOVES_TRAINING)
                        move = nn2Player.GetMoveStochastic(game, Params.NOF_SIMS_PER_MOVE_TRAINING);
                    else
                        move = nn2Player.GetMove(game, Params.NOF_SIMS_PER_MOVE_TRAINING);
                    policies.Add(nn2Player.mcts.GetPolicy());
                }
                game.DoMove(move);
                nn1Player.DoMove(move);
                nn2Player.DoMove(move);

                history.Add(move);
            }

            return game.position.score;
        }
        public int PlayOneGameGPU(List<Tuple<int, int>> history, Player aEvaluationNetworkPlayer, NeuralNetwork nn1, NeuralNetwork nn2, bool train)
        {
            Game game = new Game();
            NNPlayer nn1Player = new NNPlayer(nn1);
            NNPlayer nn2Player = new NNPlayer(nn2);

            for (int curr_ply = 0; curr_ply < GameProperties.MAXIMUM_PLYS; ++curr_ply)
            {
                if (game.IsOver())
                {
                    break;
                }
                Tuple<int, int> move = Tuple.Create<int, int>(-1, -1);

                if (game.position.sideToMove == aEvaluationNetworkPlayer)
                {
                    if (train && curr_ply < Params.STOCHASTIC_MOVES_FIRST_X_MOVES_TRAINING || !train && curr_ply < Params.STOCHASTIC_MOVES_FIRST_X_MOVES_TESTING)
                        move = nn1Player.GetMoveStochastic(game, Params.NOF_SIMS_PER_MOVE_TRAINING);
                    else
                        move = nn1Player.GetMove(game, Params.NOF_SIMS_PER_MOVE_TRAINING);
                }
                else
                {
                    if (train && curr_ply < Params.STOCHASTIC_MOVES_FIRST_X_MOVES_TRAINING || !train && curr_ply < Params.STOCHASTIC_MOVES_FIRST_X_MOVES_TESTING)
                        move = nn2Player.GetMoveStochastic(game, Params.NOF_SIMS_PER_MOVE_TRAINING);
                    else
                        move = nn2Player.GetMove(game, Params.NOF_SIMS_PER_MOVE_TRAINING);
                }
                game.DoMove(move);
                nn1Player.DoMove(move);
                nn2Player.DoMove(move);

                history.Add(move);
            }

            return game.GetScore();
        }
        public void printNNOutput(Tuple<float[], float> prediction)
        {
            for (int i = 0; i < GameProperties.GAMEBOARD_HEIGHT; ++i)
            {
                Console.WriteLine(prediction.Item1[i * GameProperties.GAMEBOARD_WIDTH + 0].ToString("0.000") + " " +
                prediction.Item1[i * GameProperties.GAMEBOARD_WIDTH + 1].ToString("0.000") + " " +
                prediction.Item1[i * GameProperties.GAMEBOARD_WIDTH + 2].ToString("0.000") + " " +
                prediction.Item1[i * GameProperties.GAMEBOARD_WIDTH + 3].ToString("0.000") + " " +
                prediction.Item1[i * GameProperties.GAMEBOARD_WIDTH + 4].ToString("0.000"));
            }
            Console.WriteLine("Value " + prediction.Item2);
        }
        private void printNNOutputs(NeuralNetwork nn)
        {
            Console.WriteLine("\nPolicy of boards");
            Game game = new Game();
            Console.WriteLine(game.position.ToString());
            Tuple<float[], float> prediction = nn.Predict(game.position);
            printNNOutput(prediction);
            startBoardValue = prediction.Item2;
            Console.WriteLine("\n");

            game.DoMove(Tuple.Create(3, 3));
            game.DoMove(Tuple.Create(0, 0));
            game.DoMove(Tuple.Create(3, 2));
            game.DoMove(Tuple.Create(4, 4));
            game.DoMove(Tuple.Create(3, 1));
            game.DoMove(Tuple.Create(3, 4));
            game.DoMove(Tuple.Create(0, 4));

            Console.WriteLine(game.position.ToString());
            prediction = nn.Predict(game.position);
            printNNOutput(prediction);
            Console.WriteLine("Note: This is a win for X");
            Console.WriteLine("\n");

            game.DoMove(Tuple.Create(1, 4));

            Console.WriteLine(game.position.ToString());
            prediction = nn.Predict(game.position);
            printNNOutput(prediction);
            Console.WriteLine("\n");

            game = new Game();

            game.DoMove(Tuple.Create(0, 0));
            game.DoMove(Tuple.Create(0, 1));
            game.DoMove(Tuple.Create(0, 2));
            game.DoMove(Tuple.Create(0, 3));
            game.DoMove(Tuple.Create(1, 0));
            game.DoMove(Tuple.Create(1, 2));
            game.DoMove(Tuple.Create(1, 3));

            Console.WriteLine(game.position.ToString());
            prediction = nn.Predict(game.position);
            printNNOutput(prediction);
            Console.WriteLine("\n");


            game = new Game();

            game.DoMove(Tuple.Create(0, 0));
            game.DoMove(Tuple.Create(4, 2));
            game.DoMove(Tuple.Create(0, 1));
            game.DoMove(Tuple.Create(4, 1));
            game.DoMove(Tuple.Create(0, 2));
            game.DoMove(Tuple.Create(3, 3));

            Console.WriteLine(game.position.ToString());
            prediction = nn.Predict(game.position);
            printNNOutput(prediction);
            Console.WriteLine("\n");


            game = new Game();

            game.DoMove(Tuple.Create(2, 0));
            game.DoMove(Tuple.Create(0, 0));
            game.DoMove(Tuple.Create(4, 2));
            game.DoMove(Tuple.Create(0, 1));
            game.DoMove(Tuple.Create(4, 1));
            game.DoMove(Tuple.Create(0, 2));
            game.DoMove(Tuple.Create(3, 3));

            Console.WriteLine(game.position.ToString());
            prediction = nn.Predict(game.position);

            printNNOutput(prediction);
            Console.WriteLine("\n");


            game = new Game();

            game.DoMove(Tuple.Create(0, 3));
            game.DoMove(Tuple.Create(2, 2));
            game.DoMove(Tuple.Create(1, 3));
            game.DoMove(Tuple.Create(3, 3));
            game.DoMove(Tuple.Create(2, 4));
            game.DoMove(Tuple.Create(4, 4));
            game.DoMove(Tuple.Create(2, 0));

            Console.WriteLine(game.position.ToString());
            prediction = nn.Predict(game.position);
            printNNOutput(prediction);
            Console.WriteLine("\n");
        }
    }
}
