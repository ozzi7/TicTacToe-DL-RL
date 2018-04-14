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
                PlayOneGame();
            }
        }
        public int PlayOneGame()
        {
            List<Tuple<int, int>> history = new List<Tuple<int, int>>();
            Game game = new Game();
            MCTSRootNode = new Node<Position>();
            MCTSRootNode.Value = game.pos;

            for (int curr_ply = 0; curr_ply < Params.maxPlies; ++curr_ply)
            {
                List<Tuple<int, int>> moves = game.GetMoves();

                if (game.HasWinner())
                {
                    return (game.pos.sideToMove == 1) ? -1 : 1;
                }
                else if (moves.Count == 0 && game.IsDrawn())
                {
                    return 0;
                }

                Tuple<int,int> move = MCTSRootNode.Value.bestMove;
                game.DoMove(move);
                history.Add(move);
                MCTSRootNode = MCTSRootNode.Value.bestChild;
            }
            return game.pos.score;
        }
        /// <summary>
        /// The Search uses a tree of game states and expands it until it runs out of time
        /// Nodes which look promising according to the NN are expanded greedily
        /// The move with highest visit count is stored in the tree node
        /// </summary>
        /// <param name="game"></param>
        /// <returns></returns>
        private double Search(Node<Position> currNode)
        {
            if(currNode.visitCount == 0)
            {
                currNode.visitCount++;
                Tuple<List<double>, double> currPosPrediction = nn.Predict(currNode.Value);
                return -currPosPrediction.Item2;
            }
            Game game = new Game(currNode.Value);
            if (!game.HasMoves())
            {
                return -game.pos.score;
            } 
                
            if (currNode.visitCount == 0)
            {
                currNode.visitCount++;
                Tuple<List<double>, double> currPosPrediction = nn.Predict(currNode.Value);
                return -currPosPrediction.Item2;
            }

            for(int i = 0; i < game.GetMoves().Count; ++i)
            {
                UCT_score = Q[s][a] + c_puct * P[s][a] * sqrt(sum(N[s])) / (1 + N[s][a])
                if u > max_u:
                    max_u = u
                    best_a = a
            }
               
            a = best_a
    
            sp = game.nextState(s, a)
            v = search(sp, game, nnet)

            Q[s][a] = (N[s][a]*Q[s][a] + v)/(N[s][a]+1)
            N[s][a] += 1
            return -v
            */
        }
    }
}
