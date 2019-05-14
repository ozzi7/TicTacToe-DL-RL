using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace TicTacToe_DL_RL
{
    class Trainer
    {
        private Player evaluationNetworkPlayer; // the network to be evaulated plays which color?
        private NeuralNetwork currentNN;
        private NeuralNetwork previousNN1;
        private NeuralNetwork previousNN2;
        private NeuralNetwork previousNN3;
        private NeuralNetwork playingNN;
        private Node<TicTacToePosition> MCTSRootNode;
        private double totalWins = 0;
        private double totalWinsAgainstRandom = 0;
        private double totalGamesAgainstRandom = 0;
        private double totalWinsX = 0;
        private double totalWinsZ = 0;
        private double totalDraws = 0;
        private double totalGames = 0;
        private double totalMoves = 0;
        private String plotFilename = "plotdata.txt";
        private double currentPseudoELO = 0;

        public Trainer() { }
        public Trainer(NeuralNetwork aCurrentNN)
        {
            currentNN = aCurrentNN;
            previousNN1 = new NeuralNetwork();
            previousNN2 = new NeuralNetwork();
            previousNN3 = new NeuralNetwork();

            previousNN1.weights = new List<float>(currentNN.weights);
            previousNN2.weights = new List<float>(currentNN.weights);
            previousNN3.weights = new List<float>(currentNN.weights);

            playingNN = new NeuralNetwork();

            if (File.Exists(plotFilename))
            {
                File.Delete(plotFilename);
            }
        }

        public void Train()
        {
            currentNN.SaveWeightsToFile("weights_start.txt");
            for (int i = 0; i < Params.nofEpochs; ++i)
            {
                TrainingRun(i);
                if(i % 100 == 0)
                    currentNN.SaveWeightsToFile("weights_net_" + ((int)(i/100)).ToString() + ".txt");
            }
        }
        public void TrainingRun(int run)
        {
            List<float> rewards = new List<float>(Params.populationSize);
            List<List<float>> weights = new List<List<float>>(Params.populationSize);
            List<List<float>> noise = new List<List<float>>(Params.populationSize);

            playingNN.weights = new List<float>(currentNN.weights);
            playingNN.ParseWeights();

            /*create noise*/
            for (int j = 0; j < Params.populationSize/2; ++j)
            {
                List<float> temp1 = new List<float>();

                for (int k = 0; k < currentNN.weights.Count; ++k)
                {
                    temp1.Add(RandomNr.GetGaussianFloat());
                }
                noise.Add(temp1);

                /* add the negative of the previous vector (so the perturbations are balanced in all directions)*/
                List<float>  temp2 = new List<float>();
                for (int k = 0; k < currentNN.weights.Count; ++k)
                {
                    temp2.Add(-temp1[k]);
                }
                noise.Add(temp2);
            }
            for (int j = 0; j < Params.populationSize; ++j)
            {
                weights.Add(new List<float>());
            }

            for (int i = 0; i < Params.populationSize; ++i)
            {
                /* create new weights for this network */
                weights[i] = new List<float>(currentNN.weights);

                for (int j = 0; j < weights[i].Count; ++j)
                {
                    weights[i][j] += Params.sigma * noise[i][j];
                }
                playingNN.weights = new List<float>(weights[i]);
                playingNN.ParseWeights();

                /* get reward of network*/
                List<Tuple<int, int>> history = new List<Tuple<int, int>>();
                float totalReward = 0;
                for (int j = 0; j < Params.gamesPerIndividuum; ++j) // if more than 2 games we need some noise
                {
                    history.Clear();
                    Player evaluationNetworkPlayer = (j % 2) == 0 ? Player.X : Player.Z;

                    Params.noiseWeight = 0.2f;
                    int result = PlayOneGame(history, evaluationNetworkPlayer, playingNN, currentNN, true);

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
                        TicTacToeGame game = new TicTacToeGame();
                        game.DisplayWinner(history);
                        if (i == Params.populationSize - 1)
                        {
                            game = new TicTacToeGame();
                            game.DisplayHistory(history);
                        }
                    }
                }

                rewards.Add(totalReward);
            }
            // debug
            //Console.WriteLine("" + string.Join(",", rewards) + "\n");
            // rewards[i] is total reward for the games of player i
            for (int i = 0; i < rewards.Count; ++i)
            {
                rewards[i] = (rewards[i] > 0) ? rewards[i] : 0; // set reward to 0 if negative
            }
            /* normalize rewards */
            float sum = rewards.Sum();
            for (int i = 0; i < rewards.Count; ++i)
            {
                rewards[i] -= sum / rewards.Count;
            }

            /* set weight for new network */
            for (int j = 0; j < currentNN.weights.Count; ++j)
            {
                float offset = 0.0f;
                for (int k = 0; k < Params.populationSize; ++k)
                {
                    offset += rewards[k] * noise[k][j];
                }
                currentNN.weights[j] += Params.alpha / (Params.populationSize * Params.sigma) * offset;
            }
            currentNN.ParseWeights();

            /* check performance vs previous best*/
            int wins = 0;
            int draws = 0;
            int losses = 0;
            for (int j = 0; j < Params.nofTestGames; ++j)
            {
                List<Tuple<int, int>> history = new List<Tuple<int, int>>();
                Player evaluationNetworkPlayer = (j % 2) == 0 ? Player.X : Player.Z;

                Params.noiseWeight = 0.2f;
                int result1 = PlayOneGame(history, evaluationNetworkPlayer, currentNN, previousNN1, false);
                //int result2 = PlayOneGame(history, evaluationNetworkPlayer, currentNN, previousNN2);
                //int result3 = PlayOneGame(history, evaluationNetworkPlayer, currentNN, previousNN3);

                totalMoves += history.Count;
                if (result1 == 1)
                {
                    totalWinsX++;
                }
                else if (result1 == -1)
                {
                    totalWinsZ++;
                }

                totalWins = totalWinsX + totalWinsZ;

                if (evaluationNetworkPlayer == Player.X && result1 == 1 ||
                    evaluationNetworkPlayer == Player.Z && result1 == -1)
                {
                    wins++;
                }
                else if (result1 == 0)
                {
                    draws++;
                }
                else
                {
                    losses++;
                }
                totalGames += 1;
            }

            totalDraws += draws;

            double winrateVsRandom = PlayAgainstRandom(20, currentNN);
            Console.WriteLine("Score: W/D/L " + wins + "/" + draws + "/" + losses + " Total winrateX/winrateZ/drawrate "+
                Math.Round(totalWinsX/totalGames,2) + "/" + Math.Round(totalWinsZ / totalGames, 2) + "/" + Math.Round(totalDraws/totalGames,2));

            if (wins < losses)
            {
                currentPseudoELO += 0;

                // ignore new network, it was badn
                currentNN.weights = new List<float>(previousNN1.weights);
                currentNN.ParseWeights();
            }
            else
            {
                currentPseudoELO += (float)(wins-losses)/ (float)Params.nofTestGames;
                //previousNN3 = previousNN2;
                //previousNN2 = previousNN1;
                previousNN1.weights = new List<float>(currentNN.weights);
                previousNN1.ParseWeights();
            }

            using (System.IO.StreamWriter file = new System.IO.StreamWriter(plotFilename, true))
            {
                file.WriteLine(currentPseudoELO + " " + Math.Round(totalWinsX / totalGames, 3) + " " +
                    Math.Round(totalWinsZ / totalGames, 3) + " " + Math.Round(totalDraws / totalGames, 3) + " " +
                    Math.Round(totalMoves/totalGames,3) + " " + Math.Round(winrateVsRandom,3));
            }
        }
        /// <summary>
        /// Play against random player
        /// </summary>
        /// <param name="nofGames"></param>
        /// <returns>Winrate</returns>
        public double PlayAgainstRandom(int nofGames, NeuralNetwork NN)
        {
            for (int j = 0; j < nofGames; ++j)
            {
                List<Tuple<int, int>> history = new List<Tuple<int, int>>();
                Player evaluationNetworkPlayer = (j % 2) == 0 ? Player.X : Player.Z;

                Params.noiseWeight = 0.0f;

                TicTacToeGame game = new TicTacToeGame();

                float result = 0.0f;
                for (int curr_ply = 0; curr_ply < Params.maxPlies; ++curr_ply)  // we always finish the game for tic tac toe
                {
                    MCTSRootNode = new Node<TicTacToePosition>(null);
                    MCTSRootNode.Value = game.pos;

                    if (game.IsOver())
                    {
                        result = game.GetScore();
                        break;
                    }

                    DirichletNoise.InitDirichlet(game.GetMoves().Count); // for root node (all root nodes not just the actual game start)
                                                                         // also tree use makes this a bit less effective going down the tree, maybe use temperature later

                    int best_child_index = -1;
                    if (game.pos.sideToMove == evaluationNetworkPlayer)
                    {
                        for (int simulation = 0; simulation < Params.nofSimsPerPosTest; ++simulation)
                        {
                            SearchUsingNN(MCTSRootNode, NN, NN); 
                        }
                        best_child_index = findBestChildWinrate(MCTSRootNode);
                    }
                    else
                    {
                        SearchUsingNN(MCTSRootNode, NN, NN); // just in case we dont create the children properly for random player
                        best_child_index = RandomNr.GetInt(0, MCTSRootNode.Children.Count);
                    }

                    List<Tuple<int, int>> moves = game.GetMoves();
                    Tuple<int, int> move = moves[best_child_index]; // add randomness here
                    game.DoMove(move);
                    history.Add(move);

                    /* tree re-use */
                    MCTSRootNode = MCTSRootNode.Children[best_child_index];
                }

                result = game.pos.score;

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

            evaluationNetworkPlayer = aEvaluationNetworkPlayer;

            for (int curr_ply = 0; curr_ply < Params.maxPlies; ++curr_ply)  // we always finish the game for tic tac toe
            {
                MCTSRootNode = new Node<TicTacToePosition>(null);
                MCTSRootNode.Value = game.pos;
            
                if (game.IsOver())
                { 
                    return game.GetScore();
                }

                DirichletNoise.InitDirichlet(game.GetMoves().Count); // for root node (all root nodes not just the actual game start)
                // also tree use makes this a bit less effective going down the tree, maybe use temperature later

                if(train)
                    for (int simulation = 0; simulation < Params.nofSimsPerPosTrain; ++simulation)
                    {
                        SearchUsingNN(MCTSRootNode, NN1, NN2); // expand tree and improve accuracy at MCTSRootNode
                        //RegularMCTSSearch(MCTSRootNode);
                        // show last simulation tree
                        if (simulation == Params.nofSimsPerPosTrain - 1 && curr_ply == 0)
                        {
                           // DisplayMCTSTree(MCTSRootNode);
                        }
                    }
                else
                    for (int simulation = 0; simulation < Params.nofSimsPerPosTest; ++simulation)
                    {
                        SearchUsingNN(MCTSRootNode, NN1, NN2); // expand tree and improve accuracy at MCTSRootNode
                                                                                        //RegularMCTSSearch(MCTSRootNode);
                                                                                        // show last simulation tree
                        if (simulation == Params.nofSimsPerPosTest - 1 && curr_ply == 0)
                        {
                            // DisplayMCTSTree(MCTSRootNode);
                        }
                    }


                int best_child_index = findBestChildWinrate(MCTSRootNode);
                //int best_child_index = findBestChildVisitCount(MCTSRootNode);

                List<Tuple<int, int>> moves = game.GetMoves();
                Tuple<int, int> move = moves[best_child_index]; // add randomness here
                game.DoMove(move);
                history.Add(move);

                /* tree re-use */
                MCTSRootNode = MCTSRootNode.Children[best_child_index];
            }

            return game.pos.score;
        }
        /// <summary>
        /// The Search uses a tree Nodes<TicTacToePosition> and expands it until it runs out of time
        /// Nodes which look promising according to the NN are expanded greedily
        /// </summary>
        /// <param name="currNode"></param>
        /// <returns>Eval</returns>
        private void SearchUsingNN(Node<TicTacToePosition> currNode, NeuralNetwork NN1, NeuralNetwork NN2)
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
                /* find value, policy */
                if (currNode.nn_policy == null)
                {
                    calculateNNOutput(currNode, NN1, NN2);
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
                    score = game.GetScore();
                }
                else
                {
                    if (currNode.nn_policy == null)
                    {
                        if (currNode.Value.sideToMove == evaluationNetworkPlayer)
                        {
                            Tuple<float[], float> prediction = NN1.Predict(currNode.Value);
                            currNode.nn_policy = new List<float>(prediction.Item1);
                            currNode.nn_value = prediction.Item2;
                        }
                        else
                        {
                            Tuple<float[], float> prediction = NN2.Predict(currNode.Value);
                            currNode.nn_policy = new List<float>(prediction.Item1);
                            currNode.nn_value = prediction.Item2;
                        }
                    }

                    score = currNode.nn_value; // [-1..1] where player 1 wins at 1 and player 2 wins at -1
                }
            }

            /* update the tree with the new score and visit counts */
            backpropagateScore(currNode, score);
        }
        private int findBestChildWinrate(Node<TicTacToePosition> currNode)
        {
            float best_winrate = float.NegativeInfinity;
            int best_child_index = -1;

            /* add dirichlet noise to root */
            if (currNode.parent == null)
            {
                for (int i = 0; i < MCTSRootNode.Children.Count; ++i)
                {
                    float winrate_temp = MCTSRootNode.Children[i].winrate*(1- Params.noiseWeight) + Params.noiseWeight * DirichletNoise.GetNoise(i);
                    if (winrate_temp > best_winrate)
                    {
                        best_winrate = winrate_temp;
                        best_child_index = i;
                    }
                }
            }
            else
            {
                for (int i = 0; i < MCTSRootNode.Children.Count; ++i)
                {
                    if (MCTSRootNode.Children[i].winrate > best_winrate)
                    {
                        best_winrate = MCTSRootNode.Children[i].winrate;
                        best_child_index = i;
                    }
                }
            }
            return best_child_index;
        }
        private int findBestChildVisitCount(Node<TicTacToePosition> currNode)
        {
            float best_visit_count = 0;
            int best_child_index = -1;

            for (int i = 0; i < MCTSRootNode.Children.Count; ++i)
            {
                if (MCTSRootNode.Children[i].visitCount > best_visit_count)
                {
                    best_visit_count = MCTSRootNode.Children[i].visitCount;
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
                    child.Value = new TicTacToePosition(game.pos);
                    child.moveIndex = moves[i].Item1 * 3 + moves[i].Item2;
                    currNode.AddChild(child);
                }
            }
        }
        private void calculateNNOutput(Node<TicTacToePosition> currNode, NeuralNetwork NN1, NeuralNetwork NN2)
        {
            if (currNode.Value.sideToMove == evaluationNetworkPlayer)
            {
                Tuple<float[], float> prediction = NN1.Predict(currNode.Value);
                currNode.nn_policy = new List<float>(prediction.Item1);
                currNode.nn_value = prediction.Item2;
            }
            else
            {
                Tuple<float[], float> prediction = NN2.Predict(currNode.Value);
                currNode.nn_policy = new List<float>(prediction.Item1);
                currNode.nn_value = prediction.Item2;
            }
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
                    float temp_UCT_score = currNode.Children[i].winrate + Params.c_puct * currNode.nn_policy[currNode.Children[i].moveIndex] *
                        (float)Math.Sqrt((Math.Log(currNode.visitCount)) / (float)(currNode.Children[i].visitCount + 1));

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
            float tempScore = (score + 1.0f)/2.0f;
            while (currNode != null)
            {
                if (currNode.Value.sideToMove == Player.X)
                {
                    currNode.winrate = (currNode.visitCount * currNode.winrate + 1.0f-tempScore) / (currNode.visitCount + 1);
                }
                else if (currNode.Value.sideToMove == Player.Z)
                {
                    currNode.winrate = (currNode.visitCount * currNode.winrate + tempScore) / (currNode.visitCount + 1);
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
                score = game.GetScore();
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

                score = simulateRandomPlayout(currNode); // [-1..1] where player 1 wins at 1 and player 2 wins at -1
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
                List<Tuple<int, int>> moves = game.GetMoves(); // valid moves

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
                // dont show empty leaves
                Console.WriteLine(node.ToString());
                foreach (Node<TicTacToePosition> child in node.Children)
                {
                    DisplayMCTSNode(child);
                }
            }
        }
    }
}
