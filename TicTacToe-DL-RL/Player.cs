using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    class RandomPlayer
    {
        static Random random = new Random();
        RandomPlayer() { }
        Tuple<int,int> Play(TicTacToeGame game)
        {
            List<Tuple<int,int>> moves = game.GetMoves();
            return moves[random.Next(0, moves.Count)];
        }
    }
}
