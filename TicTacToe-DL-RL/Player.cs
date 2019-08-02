using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    public class RandomPlayer
    {
        public Tuple<int, int> GetMove(Game game)
        {
            List<Tuple<int, int>> moves = game.GetMoves();
            return moves[RandomGen2.Next(0, moves.Count)];
        }
    }
    public class NNPlayer
    {
        public MCTS mcts;
        private NeuralNetwork nn;

        public NNPlayer(NeuralNetwork ann)
        {
            nn = ann;
            mcts = new MCTS();
        }
        public Tuple<int, int> GetMove(Game game, int nofSimsPerMove)
        {
            List<Tuple<int, int>> moves = game.GetMoves();
            mcts.Search(nn, nofSimsPerMove);

            int best_child_index = mcts.findBestChildVisitCount();
            return moves[best_child_index];
        }
        public Tuple<int, int> GetMoveStochastic(Game game, int nofSimsPerMove)
        {
            mcts.Search(nn, nofSimsPerMove);

            int best_child_index = mcts.findBestChildVisitCountStochastic();
            List<Tuple<int,int>> moves = game.GetMoves();
            return moves[best_child_index];
        }
        public void DoMove(Tuple<int,int> move)
        {
            mcts.DoMove(move);
        }
    }
    public class HumanPlayer
    {
        public Tuple<int, int> GetMove(Game game)
        {
            List<Tuple<int, int>> moves = game.GetMoves();

            while (true)
            {
                Console.Write("Enter move (f.ex.: a1): ");
                string moveString = Console.ReadLine();

                int y, x;
                if (moveString.Count() >= 2)
                {
                    char start = 'a';
                    y = moveString[0] - start;
                    int.TryParse(moveString[1].ToString(), out x); // catch.. 
                    if (0 <= y && y < GameProperties.BOARD_SIZE_Y && 0 <= x && x < GameProperties.BOARD_SIZE_X)
                        return new Tuple<int, int>(y, x);

                }
            }
        }
    }
}
