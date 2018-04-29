using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    class Trainer
    {
        private NeuralNetwork nn;
        private Node<Position> MCTSRootNode;

        public Trainer(NeuralNetwork ann)
        {
            nn = ann;
        }

        public void Train()
        {
            for (int i = 0; i < Params.nofTrainingGames; ++i)
            {
                List<Tuple<int, int>> history = new List<Tuple<int, int>>();
                int result = PlayOneGame(history);
                foreach (var tuple in history)
                {
                    Console.WriteLine("{0} - {1}", tuple.Item1.ToString(), tuple.Item2.ToString());
                }
                Console.WriteLine("Result: " + result);
            }
        }

        public Trainer() {}

        public int PlayOneGame(List<Tuple<int, int>> history)
        {     
            Game game = new Game();

            MCTSRootNode = new Node<Position>();
            MCTSRootNode.Value = game.pos;

            for (int curr_ply = 0; curr_ply < Params.maxPlies; ++curr_ply)
            {
                // we always finish the game but for other games possibly infinite cycles
                List<Tuple<int, int>> moves = game.GetMoves();

                if (game.HasWinner())
                {
                    return (game.pos.sideToMove == 1) ? -1 : 1;
                }
                else if (game.IsDrawn())
                {
                    return 0;
                }

                for (int simulations = 0; simulations < Params.nofSimsPerPos; ++simulations)
                {
                    Search(MCTSRootNode); // expand tree and improve accuracy at MCTSRootNode
                }
                Tuple<int, int> move = MCTSRootNode.Value.bestMove; // add randomness here
                game.DoMove(move);
                history.Add(move);
                MCTSRootNode = MCTSRootNode.Children[MCTSRootNode.Value.bestChildIndex];
                MCTSRootNode.Value = game.pos; // this should already exist
            }
            return game.pos.score;
        }
        /// <summary>
        /// The Search uses a tree Nodes<Position> and expands it until it runs out of time
        /// Nodes which look promising according to the NN are expanded greedily
        /// The move with highest visit is stored in each Position
        /// </summary>
        /// <param name="currNode"></param>
        /// <returns></returns>Eval
        private float Search(Node<Position> currNode)
        {
            Tuple<float[], float> currPosPrediction = nn.Predict(currNode.Value);

            Game game = new Game(currNode.Value);
            if (!game.HasMoves())
            {
                return -game.pos.score;
            }

            List<Tuple<int, int>> moves = game.GetMoves(); // valid moves

            /* Create the children if they don't exist yet */
            if (!currNode.HasChild)
            {
                for(int i = 0; i < moves.Count; ++i)
                {
                    currNode.AddChild(new Node<Position>());
                }
            }

            if (currNode.visitCount == 0)
            {
                currNode.visitCount++;

                // find best move according to NN
                float bestVal = float.NegativeInfinity;
                int bestChildIndex = -1;
                Tuple<int, int> bestMove = Tuple.Create(-1, -1);
                for (int i = 0; i < moves.Count; ++i)
                {
                    if (currPosPrediction.Item1[moves[i].Item1 * 3 + moves[i].Item2] > bestVal)
                    {
                        bestVal = currPosPrediction.Item1[moves[i].Item1 * 3 + moves[i].Item2];
                        bestMove = moves[i];
                        bestChildIndex = i;
                    }
                }
                currNode.Value.bestChildIndex = bestChildIndex;
                currNode.Value.bestMove = bestMove;
                return -currPosPrediction.Item2;
            }

            currNode.visitCount++;
            int N_a_sum = 0; // visit count of all children
            for (int i = 0; i < currNode.Children.Count; ++i)
            {
                N_a_sum += currNode.N_a[i];
            }

            currNode.UCT_score = float.NegativeInfinity;
            for (int i = 0; i < currNode.Children.Count; ++i)
            {
                float temp_UCT_score = currNode.Q_a[i] + Params.c_puct * currPosPrediction.Item1[i] *
                    (float)Math.Sqrt(N_a_sum) / (1 + currNode.Children[i].visitCount);

                if(temp_UCT_score > currNode.UCT_score)
                {
                    // new best child node
                    currNode.UCT_score = temp_UCT_score;
                    currNode.Value.bestChildIndex = i;
                    currNode.Value.bestMove = moves[i];
                }
            }

            game.DoMove(currNode.Value.bestMove);// if doesnt exist
            currNode.Children[currNode.Value.bestChildIndex].Value = game.pos; // if doesnt exist

            float v = Search(currNode.Children[currNode.Value.bestChildIndex]);

            for (int i = 0; i < moves.Count; ++i)
            {
                currNode.Q_a[i] = (currNode.N_a[i] * currNode.Q_a[i] + v) / (currNode.N_a[i] + 1);
                currNode.N_a[i] += 1;
            }
            return -v;
        }
    }
}
