using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    public class MCTS
    {
        public Node<GameState> rootNode;
        public MCTS()
        {
            rootNode = new Node<GameState>(null);
            rootNode.Value = new Game().position;
        }
        public void Search(NeuralNetwork nn, int nofSims)
        {
            createChildren(rootNode);

            if (Params.GPU_ENABLED)
                SearchGPU(nn, nofSims);
            else
                SearchCPU(nn, nofSims);
        }
        public void SearchCPU(NeuralNetwork nn, int nofSims)
        {
            createChildren(rootNode);

            if (rootNode.nn_policy == null)
            {
                calculateNNOutput(rootNode, nn);

                backpropagateScore(rootNode, rootNode.nn_value);
            }

            for (int simulation = 0; simulation < nofSims; ++simulation)
            {
                SearchUsingNN(rootNode, nn);
            }
        }
        public void SearchGPU(NeuralNetwork nn, int nofSims)
        {
            Queue<Node<GameState>> pendingNNRequests = new Queue<Node<GameState>>();

            createChildren(rootNode);

            if (rootNode.nn_policy == null)
            {
                calculateNNOutputGPU(rootNode, nn, pendingNNRequests);

                backpropagateScore(rootNode, rootNode.nn_value);
            }

            for (int simulation = 0; simulation < nofSims; ++simulation)
            {
                Tuple<float[], float> result = nn.GetResultAsync(); // try to get a result, if there is one, try to get more
                while (result != null)
                {
                    Node<GameState> nodeToUpdate = pendingNNRequests.Dequeue();
                    normalizePolicy(nodeToUpdate, result.Item1);
                    nodeToUpdate.nn_value = result.Item2;
                    nodeToUpdate.waitingForGPUPrediction = false;
                    removeVirtualLoss(rootNode, nodeToUpdate);
                    backpropagateScore(rootNode, nodeToUpdate, nodeToUpdate.nn_value);

                    result = nn.GetResultAsync(); // try to get a result, if there is one, try to get more
                }
                while (pendingNNRequests.Count > Params.MAX_PENDING_NN_EVALS)
                {
                    // if we need to wait then wait
                    result = nn.GetResultSync();
                    Node<GameState> nodeToUpdate = pendingNNRequests.Dequeue();
                    normalizePolicy(nodeToUpdate, result.Item1);
                    nodeToUpdate.nn_value = result.Item2;
                    nodeToUpdate.waitingForGPUPrediction = false;
                    removeVirtualLoss(rootNode, nodeToUpdate);
                    backpropagateScore(rootNode, nodeToUpdate, nodeToUpdate.nn_value);
                }
                SearchUsingNNGPU(rootNode, nn, pendingNNRequests);
            }

            // wait for all search results before deciding on which move to play ( because of virtual losses)
            while (pendingNNRequests.Count > 0)
            {
                // if we need to wait then wait
                Tuple<float[], float> result = nn.GetResultSync();
                Node<GameState> nodeToUpdate = pendingNNRequests.Dequeue();
                normalizePolicy(nodeToUpdate, result.Item1);
                nodeToUpdate.nn_value = result.Item2;
                nodeToUpdate.waitingForGPUPrediction = false;
                removeVirtualLoss(rootNode, nodeToUpdate);
                backpropagateScore(rootNode, nodeToUpdate, nodeToUpdate.nn_value);
            }
        }
        public List<float> GetPolicy()
        {
            // after search, record root node visits for new policy vector
            List<float> policy = Enumerable.Repeat(0.0f, GameProperties.OUTPUT_POLICIES).ToList();
            float totalVisits = 0;
            foreach (var child in rootNode.Children)
            {
                policy[child.moveIndex] = child.visits;
                totalVisits += child.visits;
            }
            foreach (var child in rootNode.Children)
            {
                policy[child.moveIndex] /= totalVisits;
                if (policy[child.moveIndex] == float.NaN ||
                    policy[child.moveIndex] == float.NegativeInfinity ||
                    policy[child.moveIndex] == float.PositiveInfinity)
                {
                    policy[child.moveIndex] = 0.0f;
                }
            }
            return policy;
        }
        public void PrintPolicy()
        {
            List<float> policy = GetPolicy();
            for (int i = 0; i < GameProperties.GAMEBOARD_HEIGHT; ++i)
            {
                Console.WriteLine(policy[i * GameProperties.GAMEBOARD_WIDTH + 0].ToString("0.000") + " " +
                policy[i * GameProperties.GAMEBOARD_WIDTH + 1].ToString("0.000") + " " +
                policy[i * GameProperties.GAMEBOARD_WIDTH + 2].ToString("0.000") + " " +
                policy[i * GameProperties.GAMEBOARD_WIDTH + 3].ToString("0.000") + " " +
                policy[i * GameProperties.GAMEBOARD_WIDTH + 4].ToString("0.000"));
            }
            Console.WriteLine("Value " + (-1)*rootNode.q_value);
        }
        public void DoMove(Tuple<int,int> move)
        {
            Game game = new Game(rootNode.Value);
            List<Tuple<int, int>> moves = game.GetMoves();
            createChildren(rootNode);

            for (int i = 0; i < moves.Count; i++)
            {
                if (moves[i].Equals(move))
                {
                    rootNode = rootNode.Children[i];
                    rootNode.parent = null;
                    break;
                }
            }
        }
        /// <summary>
        /// The Search uses a tree Nodes<GameState> and expands it
        /// Nodes which look promising according to the NN are expanded greedily
        /// </summary>
        /// <param name="currNode"></param>
        /// <returns>Eval</returns>
        public void SearchUsingNN(Node<GameState> currNode, NeuralNetwork NN)
        {
            Game game = new Game(currNode.Value);

            /* find the most promising leaf node */
            currNode = findMostPromisingLeafNode(currNode);

            /* if the leaf node is a game ending state use correct score */
            game = new Game(currNode.Value);
            if (game.IsOver() && Params.USE_REAL_TERMINAL_VALUES)
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
                    calculateNNOutput(currNode, NN);
                }

                /* update the tree with the new score and visit counts */
                backpropagateScore(currNode, currNode.nn_value);
            }
        }

        /// <summary>
        /// The Search uses a tree Nodes<GameState> and expands it
        /// Nodes which look promising according to the NN are expanded greedily
        /// </summary>
        /// <param name="currNode"></param>
        /// <returns>Eval</returns>
        private void SearchUsingNNGPU(Node<GameState> currRootNode, NeuralNetwork NN, Queue<Node<GameState>> queue)
        {
            Game game = new Game(currRootNode.Value);

            /* find the most promising leaf node */
            Node<GameState> currNode = findMostPromisingLeafNode(currRootNode);

            /* if the leaf node is a game ending state use correct score */
            game = new Game(currNode.Value);
            if (game.IsOver() && Params.USE_REAL_TERMINAL_VALUES)
            {
                backpropagateScore(currRootNode, currNode, game.GetScore());
            }
            else
            {
                if (currNode.nn_policy == null)
                {
                    calculateNNOutputGPU(currNode, NN, queue); // todo: duplicate evals
                }
                else
                {
                    backpropagateScore(currRootNode, currNode, currNode.nn_value);
                }
            }
            createChildren(currNode);
        }
        /// <summary>
        /// Loss for both players
        /// </summary>
        /// <param name="currNode"></param>
        private void propagateVirtualLoss(Node<GameState> currRootnode, Node<GameState> currNode)
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
        private void propagateVirtualLoss(Node<GameState> currNode)
        {
            while (currNode != null)
            {
                currNode.virtualVisits += 1;
                currNode.score_sum -= 1.0f;
                currNode.q_value = currNode.score_sum / (currNode.visits + currNode.virtualVisits);
                currNode = currNode.GetParent();
            }
        }
        private void removeVirtualLoss(Node<GameState> currNode)
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
        private void removeVirtualLoss(Node<GameState> currRootnode, Node<GameState> currNode)
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
        private void RegularMCTSSearch(Node<GameState> currNode)
        {
            Game game = new Game(currNode.Value);
            List<Tuple<int, int>> moves = game.GetMoves();

            /* find the most promising leaf node */
            currNode = findMostPromisingLeafNode(currNode);

            /* if the leaf node is a game ending state use correct score */
            float score = 0.0f;
            game = new Game(currNode.Value);
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
        private float simulateRandomPlayout(Node<GameState> currNode)
        {
            Game game = new Game(currNode.Value);

            while (!game.IsOver())
            {
                List<Tuple<int, int>> moves = game.GetMoves();

                game.DoMove(moves[RandomGen2.Next(0, moves.Count)]);
            }

            return game.GetScore();
        }
        private void backpropagateScore(Node<GameState> currNode, float score)
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
        private void backpropagateScore(Node<GameState> currRootnode, Node<GameState> currNode, float score)
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
        public int findBestChildVisitCountStochastic(float temperature)
        {
            List<float> visits = applyTemperature(rootNode, temperature);
            float randomNr = RandomGen2.NextFloat();

            float probabilitySum = 0.0f;
            float sumVisits = 0.0f;
            List<float> moveProbabilities = new List<float>(new float[rootNode.Children.Count]);

            foreach (var childNode in rootNode.Children)
            {
                sumVisits += childNode.visits;
            }
            for (int i = 0; i < rootNode.Children.Count; ++i)
            {
                probabilitySum += rootNode.Children[i].visits / sumVisits;
                if (probabilitySum >= randomNr)
                {
                    return i;
                }
            }
            return rootNode.Children.Count - 1;
        }
        public int findBestChildVisitCount()
        {
            List<float> visits = applyTemperature(rootNode, 0.1f);
            float bestVisitCount = -1;
            int bestChildIndex = -1;

            for (int i = 0; i < visits.Count; ++i)
            {
                float tempVisitCount = visits[i];
                if (tempVisitCount > bestVisitCount)
                {
                    bestVisitCount = tempVisitCount;
                    bestChildIndex = i;
                }
            }
            return bestChildIndex;
        }
        public void createChildren(Node<GameState> currNode)
        {
            if (!currNode.HasChild)
            {
                Game game = new Game(currNode.Value);
                List<Tuple<int, int>> moves = game.GetMoves(); // valid moves

                for (int i = 0; i < moves.Count; ++i)
                {
                    game = new Game(currNode.Value);
                    game.DoMove(moves[i]);
                    Node<GameState> child = new Node<GameState>(currNode);
                    child.Value = new GameState(game.position);
                    child.moveIndex = moves[i].Item1 * GameProperties.BOARD_SIZE_X + moves[i].Item2;
                    currNode.AddChild(child);
                }
            }
        }
        private void normalizePolicy(Node<GameState> currNode, float[] rawPolicy)
        {
            currNode.nn_policy = new List<float>(new float[rawPolicy.Length]);

            /* re-normalize policy vector */
            float sum = 0;
            foreach (var child in currNode.Children)
            {
                sum += rawPolicy[child.moveIndex];
            }

            if (sum > 0)
            {
                foreach (var child in currNode.Children)
                {
                    currNode.nn_policy[child.moveIndex] = rawPolicy[child.moveIndex] / sum;
                }
            }
            else
            {
                // shouldnt happen often
                foreach (var child in currNode.Children)
                {
                    currNode.nn_policy[child.moveIndex] = 1.0f / currNode.Children.Count;
                }
            }
        }
        private List<float> applyTemperature(Node<GameState> currNode, float temp)
        {
            List<float> visits = new List<float>();
            if (temp == 0)
            {
                float max = float.NegativeInfinity;
                float maxIndex = -1;
                for (int i = 0; i < currNode.Children.Count; ++i)
                {
                    if(currNode.Children[i].visits >= max)
                    {
                        max = currNode.Children[i].visits;
                        maxIndex = i;
                    }
                }
                for (int i = 0; i < currNode.Children.Count; ++i)
                {
                    if (i != maxIndex)
                        visits.Add(0.0f);
                    else
                        visits.Add(currNode.Children[i].visits);
                }
            }
            else
            {
                for (int i = 0; i < currNode.Children.Count; ++i)
                {
                    visits.Add((float)Math.Pow(currNode.Children[i].visits, 1.0 / temp));
                }
            }
            return visits;
        }
        private void calculateNNOutput(Node<GameState> currNode, NeuralNetwork NN)
        {
            Tuple<float[], float> prediction = NN.Predict(currNode.Value);

            currNode.nn_value = prediction.Item2;
            normalizePolicy(currNode, prediction.Item1);
        }
        /// <summary>
        /// Sends prediction request to neural network and adds the node to a queue to later fill in the result
        /// </summary>
        /// <param name="currNode"></param>
        /// <param name="NN"></param>
        /// <param name="queue"></param>
        private void calculateNNOutputGPU(Node<GameState> currNode, NeuralNetwork NN, Queue<Node<GameState>> queue)
        {
            currNode.waitingForGPUPrediction = true;
            propagateVirtualLoss(currNode);
            NN.PredictGPU(currNode.Value);
            queue.Enqueue(currNode);
        }
        private void calculateNNOutputGPU(Node<GameState> currRootnode, Node<GameState> currNode, NeuralNetwork NN, Queue<Node<GameState>> queue)
        {
            currNode.waitingForGPUPrediction = true;
            propagateVirtualLoss(currRootnode, currNode);
            NN.PredictGPU(currNode.Value);
            queue.Enqueue(currNode);
        }
        private void addDirichletNoise(Node<GameState> currNode)
        {
            if (currNode.Children.Count > 0 && currNode.nn_policy != null && Params.DIRICHLET_NOISE_WEIGHT != 0.0f)
            {
                DirichletNoise dn = new DirichletNoise(currNode.Children.Count);
                for (int i = 0; i < currNode.Children.Count; ++i)
                {
                    float noise = dn.GetNoise(i);
                    currNode.nn_policy[currNode.Children[i].moveIndex] =
                        currNode.nn_policy[currNode.Children[i].moveIndex] * (1 - Params.DIRICHLET_NOISE_WEIGHT) +
                        Params.DIRICHLET_NOISE_WEIGHT * noise;
                }
                currNode.noiseAdded = true;
            }
        }
        private Node<GameState> findMostPromisingLeafNode(Node<GameState> currNode)
        {
            bool isRootNode = true; // the current rootNode of the search tree 
            while (currNode.HasChild)
            {
                if (isRootNode && !currNode.noiseAdded)
                {
                    addDirichletNoise(currNode);
                }
                isRootNode = false;
                List<int> draws = new List<int>();
                /* create the game from the GameState */
                Game game = new Game(currNode.Value);
                List<Tuple<int, int>> moves = game.GetMoves(); // valid moves

                /* find best child node (best UCT value) to expand */
                float bestUCTScore = float.NegativeInfinity;
                int bestChildIndex = -1;

                // if nnpolicy is null then also all children have no nn output, but possibly a score from endgame position
                for (int i = 0; i < currNode.Children.Count; ++i)
                {
                    float temp_UCT_score = float.NegativeInfinity;

                    // q_value
                    float childWinrate;
                    if (currNode.Children[i].visits != 0)
                        childWinrate = currNode.Children[i].q_value;
                    else
                        childWinrate = -currNode.q_value - Params.FPU_REDUCTION;

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

                    temp_UCT_score = childWinrate + explorationTerm;

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
                    currNode = currNode.Children[draws[RandomGen2.Next(0, draws.Count)]];
                }
                else
                {
                    currNode = currNode.Children[bestChildIndex];
                }
            }
            return currNode;
        }
        private void DisplayMCTSTree(Node<GameState> roodNode)
        {
            Console.WriteLine("============================ Displaying MCTS Tree (DF order) ============================\n");
            DisplayMCTSNode(roodNode);
        }
        private void DisplayMCTSNode(Node<GameState> node)
        {
            if (node.visits != 0)
            {
                Console.WriteLine(node.ToString());
                foreach (Node<GameState> child in node.Children)
                {
                    DisplayMCTSNode(child);
                }
            }
        }
    }
}
