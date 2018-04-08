using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    class Game
    {
        public int[,] gameBoard = new int[3, 3] { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
        public int playerX = 1;
        public int playerY = -1;

        public List<Tuple<int,int>> GetMoves()
        {
            List<Tuple<int,int>> moves = new List<Tuple<int, int>>();
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (gameBoard[i, j] == 0)
                        moves.Add(Tuple.Create(i, j));
            return moves;
        }
        public bool HasMoves()
        {
            return (GetMoves().Count > 0);
        }
        public void DoMove(Tuple<int,int> move, int color)
        {
            gameBoard[move.Item1, move.Item2] = color;
        }
    }
}
