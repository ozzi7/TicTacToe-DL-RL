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

        public Trainer() {}

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
                else if (game.IsDrawn())
                {
                    return 0;
                }

                double v = Search(MCTSRootNode);
                Tuple<int,int> move = MCTSRootNode.Value.bestMove;
                game.DoMove(move);
                history.Add(move);
                MCTSRootNode = MCTSRootNode.Children[MCTSRootNode.Value.bestChildIndex];
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
            Tuple<List<float>, float> currPosPrediction = nn.Predict(currNode.Value);

            if (currNode.visitCount == 0)
            {
                currNode.visitCount++;

                return -currPosPrediction.Item2;
            }
            Game game = new Game(currNode.Value);
            if (!game.HasMoves())
            {
                return -game.pos.score;
            }

            List<Tuple<int, int>> moves = game.GetMoves();

            /* Create the children if they don't exist yet */
            if (!currNode.HasChild)
            {
                for(int i = 0; i < moves.Count; ++i)
                {
                    currNode.AddChild(new Node<Position>());
                }
            }
            int N_a_sum = 0; // visit count of all childrens
            for (int i = 0; i < moves.Count; ++i)
            {
                N_a_sum += currNode.Children[i].visitCount;
            }

            for (int i = 0; i < moves.Count; ++i)
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

            game.DoMove(currNode.Value.bestMove);
            float v = Search(currNode.Children[currNode.Value.bestChildIndex]);

            for (int i = 0; i < moves.Count; ++i)
            {
                currNode.Q_a[i] = (currNode.N_a[i] * currNode.Q_a[i] + v) / (currNode.N_a[i] + 1);
                currNode.N_a[i] += 1;
            }
            return -v;

            /*for(int i = 0; i < game.GetMoves().Count; ++i)
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
