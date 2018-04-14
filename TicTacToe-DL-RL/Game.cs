using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    class Game
    {
<<<<<<< HEAD
        public Position pos;

        public int playerX = 1;
        public int playerY = -1;

        public Game()
        {
            pos = new Position();
        }
        public Game(Position aPos)
        {
            pos = aPos;
        }
=======
        public int[,] gameBoard = new int[3, 3] { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
        public int playerX = 1;
        public int playerY = -1;
        public int sideToMove = 1;
        public int score = 0;

>>>>>>> e402893c6ab723b8426140b7615569a223867169
        public List<Tuple<int,int>> GetMoves()
        {
            List<Tuple<int,int>> moves = new List<Tuple<int, int>>();
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
<<<<<<< HEAD
                    if (pos.gameBoard[i, j] == 0)
=======
                    if (gameBoard[i, j] == 0)
>>>>>>> e402893c6ab723b8426140b7615569a223867169
                        moves.Add(Tuple.Create(i, j));
            return moves;
        }
        public bool HasMoves()
        {
            return (GetMoves().Count > 0);
        }
        public void DoMove(Tuple<int, int> move)
        {
<<<<<<< HEAD
            pos.gameBoard[move.Item1, move.Item2] = pos.sideToMove;
            pos.sideToMove = pos.sideToMove == playerX ? playerY : playerX;

            if(HasWinner())
            {
                pos.score = pos.sideToMove == playerX ? playerY : playerX;
=======
            gameBoard[move.Item1, move.Item2] = sideToMove;
            sideToMove = sideToMove == playerX ? playerY : playerX;

            if(HasWinner())
            {
                score = sideToMove == playerX ? playerY : playerX;
>>>>>>> e402893c6ab723b8426140b7615569a223867169
            }
        }
        public Tuple<int, int> GetMove()
        {
            List<Tuple<int, int>> moves = GetMoves();
            return moves[RandomNr.GetInt(0, moves.Count)];
        }
        public bool HasWinner()
        {
<<<<<<< HEAD
            if (((pos.gameBoard[0, 0] == pos.gameBoard[0, 1]) && (pos.gameBoard[0, 0] == pos.gameBoard[0, 2])) ||
               ((pos.gameBoard[1, 0] == pos.gameBoard[1, 1]) && (pos.gameBoard[1, 0] == pos.gameBoard[1, 2])) ||
               ((pos.gameBoard[2, 0] == pos.gameBoard[2, 1]) && (pos.gameBoard[2, 0] == pos.gameBoard[2, 2])) ||

               ((pos.gameBoard[0, 0] == pos.gameBoard[1, 0]) && (pos.gameBoard[0, 0] == pos.gameBoard[2, 0])) ||
               ((pos.gameBoard[0, 1] == pos.gameBoard[1, 1]) && (pos.gameBoard[0, 1] == pos.gameBoard[2, 1])) ||
               ((pos.gameBoard[0, 2] == pos.gameBoard[1, 2]) && (pos.gameBoard[0, 2] == pos.gameBoard[2, 2])) ||

               ((pos.gameBoard[0, 0] == pos.gameBoard[1, 1]) && (pos.gameBoard[0, 0] == pos.gameBoard[2, 2])) ||
               ((pos.gameBoard[0, 2] == pos.gameBoard[1, 1]) && (pos.gameBoard[0, 2] == pos.gameBoard[2, 0])))
=======
            if (((gameBoard[0, 0] == gameBoard[0, 1]) && (gameBoard[0, 0] == gameBoard[0, 2])) ||
               ((gameBoard[1, 0] == gameBoard[1, 1]) && (gameBoard[1, 0] == gameBoard[1, 2])) ||
               ((gameBoard[2, 0] == gameBoard[2, 1]) && (gameBoard[2, 0] == gameBoard[2, 2])) ||

               ((gameBoard[0, 0] == gameBoard[1, 0]) && (gameBoard[0, 0] == gameBoard[2, 0])) ||
               ((gameBoard[0, 1] == gameBoard[1, 1]) && (gameBoard[0, 1] == gameBoard[2, 1])) ||
               ((gameBoard[0, 2] == gameBoard[1, 2]) && (gameBoard[0, 2] == gameBoard[2, 2])) ||

               ((gameBoard[0, 0] == gameBoard[1, 1]) && (gameBoard[0, 0] == gameBoard[2, 2])) ||
               ((gameBoard[0, 2] == gameBoard[1, 1]) && (gameBoard[0, 2] == gameBoard[2, 0])))
>>>>>>> e402893c6ab723b8426140b7615569a223867169
                return true;
            else
                return false;
        }
        public bool IsDrawn()
        {
            return !HasMoves() && !HasWinner();
        }            
    }
<<<<<<< HEAD
    class Position
    {
        public int[,] gameBoard = new int[3, 3] { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
        public int sideToMove = 1;
        public int score = 0;

        public double UCT_score = Double.NegativeInfinity;
        public int visitCount = 0;
        public List<int> N_a = new List<int>();
        public List<int> Q_a = new List<int>();

        public Tuple<int,int> bestMove = Tuple.Create(-1,-1);
        public Node<Position> bestChild = null;

        public Position() {}
    }
=======
>>>>>>> e402893c6ab723b8426140b7615569a223867169
}
