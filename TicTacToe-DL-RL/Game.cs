using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    class Game
    {
        public Position pos;

        public int playerX = 1;
        public int playerY = -1;

        public Game()
        {
            pos = new Position();
        }
        public Game(Position aPos)
        {
            pos = new Position(aPos);
        }
        public List<Tuple<int,int>> GetMoves()
        {
            List<Tuple<int,int>> moves = new List<Tuple<int, int>>();
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    if (pos.gameBoard[i, j] == 0)
                        moves.Add(Tuple.Create(i, j));
            return moves;
        }
        public bool HasMoves()
        {
            return (GetMoves().Count > 0);
        }
        public void DoMove(Tuple<int, int> move)
        {
            pos.gameBoard[move.Item1, move.Item2] = pos.sideToMove;
            pos.sideToMove = pos.sideToMove == playerX ? playerY : playerX;

            if(HasWinner())
            {
                pos.score = pos.sideToMove == playerX ? playerY : playerX;
            }
        }
        public Tuple<int, int> GetMove()
        {
            List<Tuple<int, int>> moves = GetMoves();
            return moves[RandomNr.GetInt(0, moves.Count)];
        }
        public bool HasWinner()
        {
            if (((pos.gameBoard[0, 0] == pos.gameBoard[0, 1]) && (pos.gameBoard[0, 0] == pos.gameBoard[0, 2]) && pos.gameBoard[0, 2] != 0) ||
               ((pos.gameBoard[1, 0] == pos.gameBoard[1, 1]) && (pos.gameBoard[1, 0] == pos.gameBoard[1, 2]) && pos.gameBoard[1, 2] != 0) ||
               ((pos.gameBoard[2, 0] == pos.gameBoard[2, 1]) && (pos.gameBoard[2, 0] == pos.gameBoard[2, 2]) && pos.gameBoard[2, 2] != 0) ||

               ((pos.gameBoard[0, 0] == pos.gameBoard[1, 0]) && (pos.gameBoard[0, 0] == pos.gameBoard[2, 0]) && pos.gameBoard[2, 0] != 0) ||
               ((pos.gameBoard[0, 1] == pos.gameBoard[1, 1]) && (pos.gameBoard[0, 1] == pos.gameBoard[2, 1]) && pos.gameBoard[2, 1] != 0) ||
               ((pos.gameBoard[0, 2] == pos.gameBoard[1, 2]) && (pos.gameBoard[0, 2] == pos.gameBoard[2, 2]) && pos.gameBoard[2, 2] != 0) ||

               ((pos.gameBoard[0, 0] == pos.gameBoard[1, 1]) && (pos.gameBoard[0, 0] == pos.gameBoard[2, 2]) && pos.gameBoard[2, 2] != 0) ||
               ((pos.gameBoard[0, 2] == pos.gameBoard[1, 1]) && (pos.gameBoard[0, 2] == pos.gameBoard[2, 0]) && pos.gameBoard[2, 2] != 0))
                return true;
            else
                return false;
        }
        public bool IsDrawn()
        {
            return !HasMoves() && !HasWinner();
        }            
    }
    class Position
    {
        public int[,] gameBoard = new int[3, 3] { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
        public int sideToMove = 1;
        public int score = 0;

        public Tuple<int,int> bestMove = Tuple.Create(-1,-1);
        public int bestChildIndex = -1;

        public Position() {}
        public Position(Position aPos)
        {
            // create copy of other position
            gameBoard = aPos.gameBoard.Clone() as int[,];
            sideToMove = aPos.sideToMove;
            score = aPos.score;
        }
    }
}
