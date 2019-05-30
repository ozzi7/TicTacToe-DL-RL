using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Threading;

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
        private MovingAverage winrateVsRandMovingAvg = new MovingAverage();

        private double currentPseudoELO = 0;
        private float winrateVsRandTotal = -1;

        public Trainer(NeuralNetwork aCurrentNN)
        {
            currentNN = aCurrentNN;
            bestNN = new NeuralNetwork(currentNN.weights, currentNN.untrainable_weights);

            if (File.Exists(Params.PLOT_FILENAME))
                File.Delete(Params.PLOT_FILENAME);
            
            if (Params.GPU_ENABLED)
                OpenCL.Init(Math.Max(Params.NOF_OFFSPRING*2, Params.NOF_GAMES_TEST*2));
        }

        public void Train()
        {
            currentNN.SaveWeightsToFile("weights_start.txt");

            for (int i = 0; i < Params.NOF_EPOCHS; ++i)
            {
                TrainingRun(i);
                if (i % Params.SAVE_WEIGHT_EVERY_Xth_EPOCH == 0)
                    currentNN.SaveWeightsToFile("weights_net_" + ((int)(i / Params.SAVE_WEIGHT_EVERY_Xth_EPOCH)).ToString() + ".txt");
            }
        }
        public void TrainingRun(int run)
        {
            //################################# GENERATE NEW WEIGHTS ###########################################

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
                    nns[i].GPU_PREDICT = true;
                    currnns[i].GPU_PREDICT = true;

                    nns[i].OpenCLInit(ID.GetGlobalID());
                    currnns[i].OpenCLInit(ID.GetGlobalID());
                }
                OpenCL.CreateNetworkWeightBuffers();
            }

            // ###################################### GPU TRAINING LOOP ##############################################

            Params.DIRICHLET_NOISE_WEIGHT = 0.2f;
            if (Params.GPU_ENABLED)
            {
                Thread thread = new Thread(OpenCL.Run);
                thread.Start();

                Parallel.For(0, Params.NOF_OFFSPRING, new ParallelOptions { MaxDegreeOfParallelism = Params.MAX_THREADS_CPU }, i =>
                {
                    /* get reward of network*/
                    List<Tuple<int, int>> history = new List<Tuple<int, int>>();
                    float totalReward = 0;
                    for (int j = 0; j < Params.NOF_GAMES_PER_OFFSPRING; ++j) // if more than 2 games we need some noise
                    {
                        history.Clear();
                        Player evaluationNetworkPlayer = (j % 2) == 0 ? Player.X : Player.Z;


                        int result = PlayOneGame(history, evaluationNetworkPlayer, nns[i], currnns[i], true);

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

                        /* to display some games (debugging)*/
                        if (run % 40 == 0)
                        {
                            if (i == Params.NOF_OFFSPRING - 1 && j < 2)
                            {
                                TicTacToeGame game = new TicTacToeGame();
                                game.DisplayHistory(history);
                            }
                        }
                    }

                    rewards[i] = totalReward;
                    
                 });
                thread.Abort();
            }

            // ###################################### CPU TRAINING LOOP ##############################################

            else
            {
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

            // ########################## CREATE NEW NETWORK GIVEN REWARDS FROM TRAINING LOOP ###########################

            // rewards[i] is total reward for the games of player i
            for (int i = 0; i < rewards.Count; ++i)
            {
                rewards[i] = (rewards[i] > 0) ? rewards[i] : 0; // set reward to 0 if negative
            }
            /* normalize rewards (we could also divide by (standard deviation + eps) after subtracting the mean */
            float sum = rewards.Sum();
            for (int i = 0; i < rewards.Count; ++i)
            {
                rewards[i] -= sum / rewards.Count;
            }

            /* set weight for new network */
            for (int j = 0; j < currentNN.weights.Count; ++j)
            {
                float offset = 0.0f;
                for (int k = 0; k < Params.NOF_OFFSPRING; ++k)
                {
                    offset += rewards[k] * noise[k][j];
                }
                currentNN.weights[j] += Params.LEARNING_RATE / (Params.NOF_OFFSPRING * Params.NOISE_SIGMA) * offset;
            }
            currentNN.ParseWeights();

            // ######################## RUN TEST GAMES TO CHECK IF NEW NETWORK IS BETTER ###############################

            List<int> wins = new List<int>();
            List<int> draws = new List<int>();
            List<int> losses = new List<int>();
            List<int> movecount = new List<int>();
            List<int> winsX = new List<int>();
            List<int> winsZ = new List<int>();
            List<float> winrateVsRand = new List<float>();

            nns = new List<NeuralNetwork>();
            currnns = new List<NeuralNetwork>();

            for (int i = 0; i < Params.NOF_GAMES_TEST; ++i)
            {
                wins.Add(0);
                draws.Add(0);
                losses.Add(0);
                movecount.Add(0);
                winsX.Add(0);
                winsZ.Add(0);
                winrateVsRand.Add(0);

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
                    nns[i].GPU_PREDICT = true;
                    currnns[i].GPU_PREDICT = true;

                    nns[i].OpenCLInit(ID.GetGlobalID());
                    currnns[i].OpenCLInit(ID.GetGlobalID());
                }
                OpenCL.CreateNetworkWeightBuffers();
            }

            // #################################### GPU TEST LOOP ##########################################

            if (Params.GPU_ENABLED)
            {
                Params.DIRICHLET_NOISE_WEIGHT = 0.2f;
                Thread thread = new Thread(OpenCL.Run);
                thread.Start();

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

                    /* to display some games (debugging)*/
                    if (run % Params.SHOW_SAMPLE_MATCHES_EVERY_XTH_EPOCH == 0 && i == Params.NOF_GAMES_TEST - 1)
                    {
                        TicTacToeGame game = new TicTacToeGame();
                        game.DisplayHistory(history);
                    }
                });
                thread.Abort();
            }

            // #################################### CPU TEST LOOP ##########################################
            else
            {
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
                    if (run % Params.SHOW_SAMPLE_MATCHES_EVERY_XTH_EPOCH == 0 && i == Params.NOF_GAMES_TEST - 1)
                    {
                        TicTacToeGame game = new TicTacToeGame();
                        game.DisplayHistory(history);
                    }
                });
            }

            // #################################### PROCESS STATISTICS ##########################################

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

            Console.WriteLine("Score: W/D/L " + winsTotal + "/" + drawsTotal + "/" + lossesTotal + "  WinrateX/Drawrate/WinrateZ " +
                Math.Round(winsAsXMovingAvg.Average, 2) + "/" + Math.Round(drawsMovingAvg.Average, 2) + "/" + Math.Round(winsAsZMovingAvg.Average, 2));


            // #################################### CREATE NEW BEST NETWORK ##########################################
            if (winsTotal < lossesTotal)
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
                printValue(bestNN);
            }


            // #################################### CHECK PERFORMANCE VS RANDOM ##########################################

            if (winsTotal >= lossesTotal || winrateVsRandTotal < 0.0f)
            {
                Params.DIRICHLET_NOISE_WEIGHT = 0.0f;
                Parallel.For(0, Params.NOF_GAMES_VS_RANDOM, new ParallelOptions { MaxDegreeOfParallelism = Params.MAX_THREADS_CPU }, i =>
                {
                    NeuralNetwork currentNN = currnns[i];
                    Player evaluationNetworkPlayer = (i % 2) == 0 ? Player.X : Player.Z;
                    winrateVsRand[i] += PlayAgainstRandom(1, currentNN, evaluationNetworkPlayer);
                });
                winrateVsRandTotal = (float)winrateVsRand.Average();
                winrateVsRandMovingAvg.ComputeAverage((decimal)winrateVsRandTotal);
            }
            else
            {
                winrateVsRandMovingAvg.ComputeAverage((decimal)winrateVsRandTotal);
            }

            // #################################### WRITE PLOTTING STATISTICS ##########################################

            using (System.IO.StreamWriter file = new System.IO.StreamWriter(Params.PLOT_FILENAME, true))
            {
                file.WriteLine(currentPseudoELO + " " + Math.Round(winsAsXMovingAvg.Average, 2) + " " +
                    Math.Round(winsAsZMovingAvg.Average, 2) + " " + Math.Round(drawsMovingAvg.Average, 2) + " " +
                    Math.Round(averageMovesMovingAvg.Average, 2) + " " + Math.Round(winrateVsRandMovingAvg.Average, 2));
            }
        }
        /// <summary>
        /// Play against random player
        /// </summary>
        /// <param name="nofGames"></param>
        /// <returns>Winrate</returns>
        public float PlayAgainstRandom(int nofGames, NeuralNetwork NN, Player evaluationNetworkPlayer)
        {
            float totalWinsAgainstRandom = 0;
            float totalGamesAgainstRandom = 0;
            for (int j = 0; j < nofGames; ++j)
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
                        break;
                    }

                    DirichletNoise dn = new DirichletNoise(game.GetMoves().Count);
                    // for root node (all root nodes not just the actual game start)
                    // also tree use makes this a bit less effective going down the tree, maybe use temperature later

                    int best_child_index = -1;
                    if (game.position.sideToMove == evaluationNetworkPlayer)
                    {
                        for (int simulation = 0; simulation < Params.NOF_SIMS_PER_MOVE_TRAINING; ++simulation)
                        {
                            SearchUsingNN(MCTSRootNode, NN);
                        }
                        best_child_index = findBestChildWinrate(MCTSRootNode, dn, curr_ply);
                        //best_child_index = findBestChildVisitCount(MCTSRootNode, dn);
                    }
                    else
                    {
                        SearchUsingNN(MCTSRootNode, NN); // just in case we dont create the children properly for random player
                        best_child_index = RandomNr.GetInt(0, MCTSRootNode.Children.Count);
                    }

                    List<Tuple<int, int>> moves = game.GetMoves();
                    Tuple<int, int> move = moves[best_child_index]; // add randomness here
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
                    totalWinsAgainstRandom++;
                }
                totalGamesAgainstRandom++;
            }
            return totalWinsAgainstRandom / totalGamesAgainstRandom;
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
                if (train)
                {
                    for (int simulation = 0; simulation < Params.NOF_SIMS_PER_MOVE_TESTING; ++simulation)
                    {
                        if(curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                            SearchUsingNN(MCTSRootNodeNN1, NN1); // expand tree and improve accuracy at MCTSRootNode
                        else
                            SearchUsingNN(MCTSRootNodeNN2, NN2); // expand tree and improve accuracy at MCTSRootNode
                        //RegularMCTSSearch(MCTSRootNode);
                        // show last simulation tree
                        if (simulation == Params.NOF_SIMS_PER_MOVE_TESTING - 1 && curr_ply == 0)
                        {
                            // DisplayMCTSTree(MCTSRootNode);
                        }
                    }

                }
                else
                {
                    for (int simulation = 0; simulation < Params.NOF_SIMS_PER_MOVE_TRAINING; ++simulation)
                    {
                        if (curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                            SearchUsingNN(MCTSRootNodeNN1, NN1); // expand tree and improve accuracy at MCTSRootNode
                        else
                            SearchUsingNN(MCTSRootNodeNN2, NN2); // expand tree and improve accuracy at MCTSRootNode

                        if (simulation == Params.NOF_SIMS_PER_MOVE_TRAINING - 1 && curr_ply == 0)
                        {
                            //DisplayMCTSTree(MCTSRootNode);
                        }
                    }
                }
                int best_child_index = -1;
                if (curr_ply % 2 == 0 && aEvaluationNetworkPlayer == Player.X || curr_ply % 2 == 1 && aEvaluationNetworkPlayer == Player.Z)
                    best_child_index = findBestChildWinrate(MCTSRootNodeNN1, dn, curr_ply);
                else
                    best_child_index = findBestChildWinrate(MCTSRootNodeNN2, dn, curr_ply);

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
                score = (game.GetScore()+1.0f)/2.0f;
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
                int best_policy_child_index = -1;
                float best_policy = float.NegativeInfinity;
                for (int i = 0; i < currNode.Children.Count; ++i)
                {
                    if (best_policy < currNode.nn_policy[currNode.Children[i].moveIndex])
                    {
                        best_policy = currNode.nn_policy[currNode.Children[i].moveIndex];
                        best_policy_child_index = i;
                    }
                }

                /*get the value of best child..*/
                currNode = currNode.Children[best_policy_child_index];

                game = new TicTacToeGame(currNode.Value);
                if (game.IsOver())
                {
                    score = (game.GetScore()+1.0f)/2.0f; // [-1, 1] where player X wins at 1 and player Z at -1
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
        private void printPolicy(NeuralNetwork nn)
        {
            Console.WriteLine("\nPolicy of empty board");
            TicTacToeGame game = new TicTacToeGame();
            Tuple<float[], float> prediction = nn.Predict(game.position);

            for (int i = 0; i < 5; ++i)
            { 
                Console.WriteLine(prediction.Item1[i * 5 + 0].ToString("0.00") + " " +
                prediction.Item1[i * 5 + 1].ToString("0.00") + " " +
                prediction.Item1[i * 5 + 2].ToString("0.00") + " " +
                prediction.Item1[i * 5 + 3].ToString("0.00") + " " +
                prediction.Item1[i * 5 + 4].ToString("0.00"));
            }
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
        private int findBestChildWinrate(Node<TicTacToePosition> currNode, DirichletNoise dn, int depth)
        {
            float best_winrate = float.NegativeInfinity;
            int best_child_index = -1;

            /* add dirichlet noise to root */
            for (int i = 0; i < currNode.Children.Count; ++i)
            {
                //float noiseWeight = ((25-depth) / 25.0f)* ((25 - depth) / 25.0f) * Params.DIRICHLET_NOISE_WEIGHT; // quadratic
                //float noiseWeight = ((25 - depth) / 25.0f) * Params.DIRICHLET_NOISE_WEIGHT; // linear
                float noiseWeight = Params.DIRICHLET_NOISE_WEIGHT; // constant
                float winrate_temp = currNode.Children[i].winrate * (1 - noiseWeight) + noiseWeight * dn.GetNoise(i);
                if (winrate_temp > best_winrate)
                {
                    best_winrate = winrate_temp;
                    best_child_index = i;
                }
            }

            return best_child_index;
        }
        private int findBestChildVisitCount(Node<TicTacToePosition> currNode, DirichletNoise dn)
        {
            float best_visit_count = 0;
            int best_child_index = -1;

            for (int i = 0; i < currNode.Children.Count; ++i)
            {
                float tempvisitCount = currNode.Children[i].visitCount * (1 - Params.DIRICHLET_NOISE_WEIGHT) + Params.DIRICHLET_NOISE_WEIGHT * dn.GetNoise(i);
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
        private Node<TicTacToePosition> findMostPromisingLeafNode(Node<TicTacToePosition> currNode)
        {
            while (currNode.HasChild)
            {
                /* create the game from the TicTacToePosition */
                TicTacToeGame game = new TicTacToeGame(currNode.Value);
                List<Tuple<int, int>> moves = game.GetMoves(); // valid moves

                /* find best child node (best UCT value) to expand */
                float bestUCTScore = float.NegativeInfinity;

                for (int i = 0; i < currNode.Children.Count; ++i)
                {
                    // get UCT of each child (we never save this for now)
                    //                float secondTerm = Params.c_puct * currNode.nn_policy[currNode.Children[i].moveIndex] *
                    //(float)Math.Sqrt(currNode.visitCount) / (float)(currNode.Children[i].visitCount + 1);
                    //                float temp_UCT_score = currNode.Children[i].winrate + secondTerm;

                    float temp_UCT_score = currNode.Children[i].winrate + Params.C_PUCT * currNode.nn_policy[currNode.Children[i].moveIndex] *
                        (float)Math.Sqrt((Math.Log(currNode.visitCount+1)) / (float)(currNode.Children[i].visitCount + 1));

                    if (temp_UCT_score > bestUCTScore)
                    {
                        // new best child 
                        currNode.bestChildIndex = i;
                        currNode.bestMove = moves[i];
                        bestUCTScore = temp_UCT_score;
                    }
                }
                game.DoMove(currNode.bestMove);
                currNode = currNode.Children[currNode.bestChildIndex];
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
                    //currNode.winrate = (currNode.visitCount * currNode.winrate + score) / (currNode.visitCount + 1);
                }
                else if (currNode.Value.sideToMove == Player.Z)
                {
                    currNode.winrate = (currNode.visitCount * currNode.winrate + score) / (currNode.visitCount + 1);
                    //currNode.winrate = (currNode.visitCount * currNode.winrate + 1.0f -score) / (currNode.visitCount + 1);
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
                int best_policy_child_index = RandomNr.GetInt(0, currNode.Children.Count);

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

                game.DoMove(moves[RandomNr.GetInt(0, moves.Count)]);
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
