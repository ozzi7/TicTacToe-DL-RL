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
<<<<<<< HEAD
            Train();
        }
        public void Train()
        {
            NeuralNetwork nn = new NeuralNetwork();
            Trainer trainer = new Trainer(nn);
            trainer.Train();
            nn.SaveToFile("test.txt");
=======
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
>>>>>>> e402893c6ab723b8426140b7615569a223867169
        }
    }
}
