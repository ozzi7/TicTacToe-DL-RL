using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    class Program
    {
        void Main(string[] args)
        {
            NeuralNetwork nn = new NeuralNetwork();
            Game game = new Game();
            Train();
        }
        void Train()
        {
            Trainer trainer = new Trainer();
            GenerateTrainingGames(trainer);
        }
        public void GenerateTrainingGames(Trainer trainer)
        {
            for (int i = 0; i < Params.nofTrainingGames; ++i)
            {
                trainer.PlayOneGame();
            }
        }
    }
}
