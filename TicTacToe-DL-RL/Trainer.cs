using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    class Trainer
    {
        public Trainer()
        {

        }

        public int PlayOneGame()
        {
            List<Tuple<int, int>> history = new List<Tuple<int, int>>();
            Game game = new Game();
            for (int curr_ply = 0; curr_ply < Params.maxPlies; ++curr_ply)
            {
                List<Tuple<int, int>> moves = game.GetMoves();

                if (game.HasWinner())
                {
                    return (game.sideToMove == 1) ? -1 : 1;
                }
                else if (moves.Count == 0 && game.IsDrawn())
                {
                    return 0;
                }

                Tuple<int,int> move = game.GetMove();
                game.DoMove(move);
                history.Add(move);
            }
            return game.score;
        }
        private void Search()
        {
            /*
             * def search(s, game, nnet):
    if game.gameEnded(s): return -game.gameReward(s)

    if s not in visited:
        visited.add(s)
        P[s], v = nnet.predict(s)
        return -v
  
    max_u, best_a = -float("inf"), -1
    for a in range(game.getValidActions(s)):
        u = Q[s][a] + c_puct*P[s][a]*sqrt(sum(N[s]))/(1+N[s][a])
        if u>max_u:
            max_u = u
            best_a = a
    a = best_a
    
    sp = game.nextState(s, a)
    v = search(sp, game, nnet)

    Q[s][a] = (N[s][a]*Q[s][a] + v)/(N[s][a]+1)
    N[s][a] += 1
    return -v
    */
        }
    }
}
