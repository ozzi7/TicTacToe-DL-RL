using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    public enum Player { X, Z }; // X always starts, X leads to winrate 1, Z to winrate -1, draw is 0

    class TicTacToeGame
    {
        public TicTacToePosition pos;

        public TicTacToeGame()
        {
            pos = new TicTacToePosition();
        }
        public TicTacToeGame(TicTacToePosition aPos)
        {
            pos = new TicTacToePosition(aPos);
        }
        public List<Tuple<int, int>> GetMoves()
        {
            List<Tuple<int, int>> moves = new List<Tuple<int, int>>();
            for (int i = 0; i < 5; ++i)
                for (int j = 0; j < 5; ++j)
                    if (pos.gameBoard[i, j] == 0)
                        moves.Add(Tuple.Create(i, j));
            return moves;
        }
        public int GetScore()
        {
            return pos.score;
        }
        /// <summary>
        /// Check if the game is in a finished TicTacToePosition (draw or win)
        /// </summary>
        /// <returns></returns>
        public bool IsOver()
        {
            return (HasWinner() || GetMoves().Count == 0);
        }
        public void DoMove(Tuple<int, int> move)
        {
            pos.gameBoard[move.Item1, move.Item2] = pos.sideToMove == Player.X ? 1 : -1;
            pos.sideToMove = pos.sideToMove == Player.X ? Player.Z : Player.X;

            if (HasWinner())
            {
                pos.score = pos.sideToMove == Player.X ? -1 : 1;
            }
            if (IsDrawn())
            {
                pos.score = 0;
            }
        }

        /// <summary>
        /// Check if the game is won by a player
        /// </summary>
        /// <returns></returns>
        public bool HasWinner()
        {
            for (int row = 0; row < 5; ++row)
            {
                if ((pos.gameBoard[row, 0] == pos.gameBoard[row, 1]) && (pos.gameBoard[row, 1] == pos.gameBoard[row, 2]) &&
                    (pos.gameBoard[row, 2] == pos.gameBoard[row, 3]) && pos.gameBoard[row, 0] != 0)
                    return true;
            
                if ((pos.gameBoard[row, 1] == pos.gameBoard[row, 2]) && (pos.gameBoard[row, 2] == pos.gameBoard[row, 3]) &&
                    (pos.gameBoard[row, 3] == pos.gameBoard[row, 4]) && pos.gameBoard[row, 1] != 0)
                    return true;
            }
            for (int col = 0; col < 5; ++col)
            {
                if ((pos.gameBoard[0, col] == pos.gameBoard[1, col]) && (pos.gameBoard[1,col] == pos.gameBoard[2, col]) &&
                    (pos.gameBoard[2, col] == pos.gameBoard[3, col]) && pos.gameBoard[0,col] != 0)
                    return true;

                if ((pos.gameBoard[1, col] == pos.gameBoard[2, col]) && (pos.gameBoard[2, col] == pos.gameBoard[3, col]) &&
                    (pos.gameBoard[3, col] == pos.gameBoard[4, col]) && pos.gameBoard[1, col] != 0)
                    return true;
            }
            if ((pos.gameBoard[0, 0] == pos.gameBoard[1, 1]) && (pos.gameBoard[1, 1] == pos.gameBoard[2, 2]) &&
                (pos.gameBoard[2, 2] == pos.gameBoard[3, 3]) && pos.gameBoard[0, 0] != 0)
                return true;
            if ((pos.gameBoard[1, 1] == pos.gameBoard[2, 2]) && (pos.gameBoard[2, 2] == pos.gameBoard[3, 3]) &&
                (pos.gameBoard[3, 3] == pos.gameBoard[4, 4]) && pos.gameBoard[1, 1] != 0)
                return true;
            if ((pos.gameBoard[0, 4] == pos.gameBoard[1, 3]) && (pos.gameBoard[1, 3] == pos.gameBoard[2, 2]) &&
                (pos.gameBoard[2, 2] == pos.gameBoard[3, 1]) && pos.gameBoard[0, 4] != 0)
                return true;
            if ((pos.gameBoard[1, 3] == pos.gameBoard[2, 2]) && (pos.gameBoard[2, 2] == pos.gameBoard[3, 1]) &&
                (pos.gameBoard[3, 1] == pos.gameBoard[4, 0]) && pos.gameBoard[1, 3] != 0)
                return true;
            if ((pos.gameBoard[0, 1] == pos.gameBoard[1, 2]) && (pos.gameBoard[1, 2] == pos.gameBoard[2, 3]) &&
                (pos.gameBoard[2, 3] == pos.gameBoard[3, 4]) && pos.gameBoard[0, 1] != 0)
                return true;
            if ((pos.gameBoard[1, 0] == pos.gameBoard[2, 1]) && (pos.gameBoard[2, 1] == pos.gameBoard[3, 2]) &&
                (pos.gameBoard[3, 2] == pos.gameBoard[4, 3]) && pos.gameBoard[1, 0] != 0)
                return true;
            if ((pos.gameBoard[0, 3] == pos.gameBoard[1, 2]) && (pos.gameBoard[1, 2] == pos.gameBoard[2, 1]) &&
                (pos.gameBoard[2, 1] == pos.gameBoard[3, 0]) && pos.gameBoard[0, 3] != 0)
                return true;
            if ((pos.gameBoard[1, 4] == pos.gameBoard[2, 3]) && (pos.gameBoard[2, 3] == pos.gameBoard[3, 2]) &&
                (pos.gameBoard[3, 2] == pos.gameBoard[4, 1]) && pos.gameBoard[1, 4] != 0)
                return true;
            return false;
        }
        /// <summary>
        /// Check if the game is drawn
        /// </summary>
        /// <returns></returns>
        public bool IsDrawn()
        {
            return IsOver() && !HasWinner();
        }
        /// <summary>
        /// Replay the game given the complete move history and display the boards during the game
        /// </summary>
        /// <param name="history"></param>
        public void DisplayHistory(List<Tuple<int, int>> history)
        {
            Console.WriteLine("\nPlayed game:");

            foreach (var move in history)
            {
                DoMove(move);
                Console.WriteLine(pos.ToString());
            }
            Console.WriteLine("Winner: " + pos.score.ToString().Replace("-1", "Z").Replace("1", "X").Replace("0","Draw") + "\n");
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
            Console.WriteLine("Winner: " + pos.score.ToString().Replace("-1", "Z").Replace("1", "X").Replace("0", "Draw"));
        }
    }
    /// <summary>
    /// Stores a complete state of the game
    /// </summary>
    class TicTacToePosition
    {
        public int[,] gameBoard = new int[5, 5] { { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 },
        { 0, 0, 0, 0, 0 }, { 0, 0, 0, 0, 0 }};
        public Player sideToMove = Player.X;
        public int score = 0; // 0 is none/draw, 1 player X, -1 player Z

        public TicTacToePosition() {}
        public TicTacToePosition(TicTacToePosition aPos)
        {
            // create copy of other TicTacToePosition
            gameBoard = aPos.gameBoard.Clone() as int[,];
            sideToMove = aPos.sideToMove;
            score = aPos.score;
        }
        public override string ToString() {
            String returnString = "Side to move: " + sideToMove + "\n";
            returnString += "Board score: " + score + "\n\n";

            String boardString =
                gameBoard[0, 0] + " " + gameBoard[1, 0] + " " + gameBoard[2, 0] + " " + gameBoard[3, 0] + " " + gameBoard[4, 0] + "\n" +
                gameBoard[0, 1] + " " + gameBoard[1, 1] + " " + gameBoard[2, 1] + " " + gameBoard[3, 1] + " " + gameBoard[4, 1] + "\n" +
                gameBoard[0, 2] + " " + gameBoard[1, 2] + " " + gameBoard[2, 2] + " " + gameBoard[3, 2] + " " + gameBoard[4, 2] + "\n" +
                gameBoard[0, 3] + " " + gameBoard[1, 3] + " " + gameBoard[2, 3] + " " + gameBoard[3, 3] + " " + gameBoard[4, 3] + "\n" +
                gameBoard[0, 4] + " " + gameBoard[1, 4] + " " + gameBoard[2, 4] + " " + gameBoard[3, 4] + " " + gameBoard[4, 4] + "\n";
            boardString = boardString.Replace("-1", "Z");
            boardString = boardString.Replace("1", "X");

            return returnString + boardString + "\n";
        }
    }
}
