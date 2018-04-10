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
        public int sideToMove = 1;
        public int score = 0;

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
        public void DoMove(Tuple<int, int> move)
        {
            gameBoard[move.Item1, move.Item2] = sideToMove;
            sideToMove = sideToMove == playerX ? playerY : playerX;

            if(HasWinner())
            {
                score = sideToMove == playerX ? playerY : playerX;
            }
        }
        public Tuple<int, int> GetMove()
        {
            List<Tuple<int, int>> moves = GetMoves();
            return moves[RandomNr.GetInt(0, moves.Count)];
        }
        public bool HasWinner()
        {
            if (((gameBoard[0, 0] == gameBoard[0, 1]) && (gameBoard[0, 0] == gameBoard[0, 2])) ||
               ((gameBoard[1, 0] == gameBoard[1, 1]) && (gameBoard[1, 0] == gameBoard[1, 2])) ||
               ((gameBoard[2, 0] == gameBoard[2, 1]) && (gameBoard[2, 0] == gameBoard[2, 2])) ||

               ((gameBoard[0, 0] == gameBoard[1, 0]) && (gameBoard[0, 0] == gameBoard[2, 0])) ||
               ((gameBoard[0, 1] == gameBoard[1, 1]) && (gameBoard[0, 1] == gameBoard[2, 1])) ||
               ((gameBoard[0, 2] == gameBoard[1, 2]) && (gameBoard[0, 2] == gameBoard[2, 2])) ||

               ((gameBoard[0, 0] == gameBoard[1, 1]) && (gameBoard[0, 0] == gameBoard[2, 2])) ||
               ((gameBoard[0, 2] == gameBoard[1, 1]) && (gameBoard[0, 2] == gameBoard[2, 0])))
                return true;
            else
                return false;
        }
        public bool IsDrawn()
        {
            return !HasMoves() && !HasWinner();
        }            
    }
}
