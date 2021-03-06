﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    // X always starts, X leads to q_value 1, Z to q_value -1 win X is 1, draw 0, win Z -1 
    // Value head in NN is -1 for win Z, 0 for draw and 1 for win X
    public enum Player { X, Z };
    public static class GameProperties
    {
        public const int GAMEBOARD_WIDTH = 5;
        public const int GAMEBOARD_HEIGHT = 5;
        public const int INPUT_PLANES = 3;
        public const int OUTPUT_POLICIES = 25;
        public const int MOVES_IN_START_POS = 25;

        public const int MAXIMUM_PLYS = 1337; // meaning the game is always finished to the end (>= 25)
        public const int BOARD_SIZE_Y = 5;
        public const int BOARD_SIZE_X = 5;
    }
    public class Game
    {
        public GameState position;

        public Game()
        {
            position = new GameState();
        }
        public Game(GameState aPos)
        {
            position = new GameState(aPos);
        }
        public List<Tuple<int, int>> GetMoves()
        {
            List<Tuple<int, int>> moves = new List<Tuple<int, int>>();
            if (!IsOver())
            {
                for (int i = 0; i < GameProperties.BOARD_SIZE_Y; ++i)
                    for (int j = 0; j < GameProperties.BOARD_SIZE_X; ++j)
                        if (position.gameBoard[i, j] == 0)
                            moves.Add(Tuple.Create(i, j));
            }
            return moves;
        }
        public int GetScore()
        {
            return position.score;
        }
        /// <summary>
        /// Check if the game is in a finished GameState (draw or win)
        /// </summary>
        /// <returns></returns>
        public bool IsOver()
        {
            return (hasWinner() || isDraw());
        }
        /// <summary>
        /// Check if the gameboard is fully occupied (draw)
        /// </summary>
        /// <returns></returns>
        private bool isDraw()
        {
            for(int i = 0; i < GameProperties.BOARD_SIZE_Y; ++i)
            {
                for(int j = 0; j < GameProperties.BOARD_SIZE_X; ++j)
                {
                    if(position.gameBoard[i,j] == 0)
                    {
                        return false;
                    }
                }
            }
            return true;
        }
        public void DoMove(Tuple<int, int> move)
        {
            position.gameBoard[move.Item1, move.Item2] = position.sideToMove == Player.X ? 1 : -1;
            position.sideToMove = position.sideToMove == Player.X ? Player.Z : Player.X;

            if (hasWinner())
            {
                position.score = position.sideToMove == Player.X ? -1 : 1;
            }
            else
            {
                position.score = 0;
            }
        }

        /// <summary>
        /// Check if the game is won by a player
        /// </summary>
        /// <returns></returns>
        private bool hasWinner()
        {
            // check rows
            for (int row = 0; row < 5; ++row)
            {
                if ((position.gameBoard[row, 0] == position.gameBoard[row, 1]) && (position.gameBoard[row, 1] == position.gameBoard[row, 2]) &&
                    (position.gameBoard[row, 2] == position.gameBoard[row, 3]) && position.gameBoard[row, 0] != 0)
                    return true;
            
                if ((position.gameBoard[row, 1] == position.gameBoard[row, 2]) && (position.gameBoard[row, 2] == position.gameBoard[row, 3]) &&
                    (position.gameBoard[row, 3] == position.gameBoard[row, 4]) && position.gameBoard[row, 1] != 0)
                    return true;
            }
            // check cols
            for (int col = 0; col < 5; ++col)
            {
                if ((position.gameBoard[0, col] == position.gameBoard[1, col]) && (position.gameBoard[1,col] == position.gameBoard[2, col]) &&
                    (position.gameBoard[2, col] == position.gameBoard[3, col]) && position.gameBoard[0,col] != 0)
                    return true;

                if ((position.gameBoard[1, col] == position.gameBoard[2, col]) && (position.gameBoard[2, col] == position.gameBoard[3, col]) &&
                    (position.gameBoard[3, col] == position.gameBoard[4, col]) && position.gameBoard[1, col] != 0)
                    return true;
            }
            // check diags
            if ((position.gameBoard[0, 0] == position.gameBoard[1, 1]) && (position.gameBoard[1, 1] == position.gameBoard[2, 2]) &&
                (position.gameBoard[2, 2] == position.gameBoard[3, 3]) && position.gameBoard[0, 0] != 0)
                return true;
            if ((position.gameBoard[1, 1] == position.gameBoard[2, 2]) && (position.gameBoard[2, 2] == position.gameBoard[3, 3]) &&
                (position.gameBoard[3, 3] == position.gameBoard[4, 4]) && position.gameBoard[1, 1] != 0)
                return true;
            if ((position.gameBoard[0, 4] == position.gameBoard[1, 3]) && (position.gameBoard[1, 3] == position.gameBoard[2, 2]) &&
                (position.gameBoard[2, 2] == position.gameBoard[3, 1]) && position.gameBoard[0, 4] != 0)
                return true;
            if ((position.gameBoard[1, 3] == position.gameBoard[2, 2]) && (position.gameBoard[2, 2] == position.gameBoard[3, 1]) &&
                (position.gameBoard[3, 1] == position.gameBoard[4, 0]) && position.gameBoard[1, 3] != 0)
                return true;
            if ((position.gameBoard[0, 1] == position.gameBoard[1, 2]) && (position.gameBoard[1, 2] == position.gameBoard[2, 3]) &&
                (position.gameBoard[2, 3] == position.gameBoard[3, 4]) && position.gameBoard[0, 1] != 0)
                return true;
            if ((position.gameBoard[1, 0] == position.gameBoard[2, 1]) && (position.gameBoard[2, 1] == position.gameBoard[3, 2]) &&
                (position.gameBoard[3, 2] == position.gameBoard[4, 3]) && position.gameBoard[1, 0] != 0)
                return true;
            if ((position.gameBoard[0, 3] == position.gameBoard[1, 2]) && (position.gameBoard[1, 2] == position.gameBoard[2, 1]) &&
                (position.gameBoard[2, 1] == position.gameBoard[3, 0]) && position.gameBoard[0, 3] != 0)
                return true;
            if ((position.gameBoard[1, 4] == position.gameBoard[2, 3]) && (position.gameBoard[2, 3] == position.gameBoard[3, 2]) &&
                (position.gameBoard[3, 2] == position.gameBoard[4, 1]) && position.gameBoard[1, 4] != 0)
                return true;
            return false;
        }

        /// <summary>
        /// Replay the game given the complete move history and display the boards during the game
        /// </summary>
        /// <param name="history"></param>
        public void DisplayHistory(List<Tuple<int, int>> history)
        {
            foreach (var move in history)
            {
                DoMove(move);
                Console.WriteLine(position.ToString());
                Console.WriteLine("################################################################################\n");
            }
            
            Console.WriteLine("Winner: " + position.score.ToString().Replace("-1", "Z").Replace("1", "X").Replace("0","Draw") + "\n");
        }
        /// <summary>
        /// Show the winner of the game given a complete history of the moves
        /// </summary>
        /// <param name="history"></param>
        public void DisplayWinner(List<Tuple<int, int>> history)
        {
            foreach (var move in history)
            {
                DoMove(move);
            }
            Console.WriteLine("Winner: " + position.score.ToString().Replace("-1", "Z").Replace("1", "X").Replace("0", "Draw"));
        }
    }
    /// <summary>
    /// Stores a complete state of the game
    /// </summary>
    public class GameState
    {
        // [Y coord, X coord]
        public int[,] gameBoard = new int[5, 5] { { 0, 0, 0, 0, 0 }, 
                                                  { 0, 0, 0, 0, 0 }, 
                                                  { 0, 0, 0, 0, 0 },
                                                  { 0, 0, 0, 0, 0 },
                                                  { 0, 0, 0, 0, 0 }};
        public Player sideToMove = Player.X;

        // 0 is none/draw, 1 player X (always starts), -1 player Z
        public int score = 0;

        public GameState() {}
        /// <summary>
        /// Create a copy of a position
        /// </summary>
        /// <param name="aPosition"></param>
        public GameState(GameState aPosition)
        {
            //gameBoard = aPosition.gameBoard.Clone() as int[,];
            for (int i = 0; i < GameProperties.BOARD_SIZE_Y; ++i)
            {
                for (int j = 0; j < GameProperties.BOARD_SIZE_X; ++j)
                {
                    gameBoard[i, j] = aPosition.gameBoard[i, j];
                }
            }
            sideToMove = aPosition.sideToMove;
            score = aPosition.score;
        }
        public override string ToString() {
            String returnString = "Side to move: " + sideToMove + "\n";
            returnString += "Board score: " + score + "\n\n";

            String boardString = 
                "a " + gameBoard[0, 0] + " " + gameBoard[0, 1] + " " + gameBoard[0, 2] + " " + gameBoard[0, 3] + " " + gameBoard[0, 4] + "\n" +
                "b " + gameBoard[1, 0] + " " + gameBoard[1, 1] + " " + gameBoard[1, 2] + " " + gameBoard[1, 3] + " " + gameBoard[1, 4] + "\n" +
                "c " + gameBoard[2, 0] + " " + gameBoard[2, 1] + " " + gameBoard[2, 2] + " " + gameBoard[2, 3] + " " + gameBoard[2, 4] + "\n" +
                "d " + gameBoard[3, 0] + " " + gameBoard[3, 1] + " " + gameBoard[3, 2] + " " + gameBoard[3, 3] + " " + gameBoard[3, 4] + "\n" +
                "e " + gameBoard[4, 0] + " " + gameBoard[4, 1] + " " + gameBoard[4, 2] + " " + gameBoard[4, 3] + " " + gameBoard[4, 4] + "\n";
            boardString = boardString.Replace("-1", "Z");
            boardString = boardString.Replace("1", "X");
            boardString = boardString.Replace("0", ".");
            boardString = "  0 1 2 3 4\n" + boardString;
            return returnString + boardString + "\n";
        }
        public override int GetHashCode()
        {
            int hash = 0;
            int index = 1;
            for (int i = 0; i < GameProperties.BOARD_SIZE_Y; ++i)
            {
                for (int j = 0; j < GameProperties.BOARD_SIZE_X; ++j)
                {
                    hash = hash + (gameBoard[i, j]*(int)Math.Pow(index,2)); // convert -1,0,1 to 0,1,2
                }
            }
            hash = sideToMove == Player.X ? hash : hash *(-1);
            return hash;
        }
    }
}
