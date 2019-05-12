/*
 * Used for MCTS
 * Nodes have a board TicTacToePosition and some metadata such as upper confidence winrate
*/
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    public class Node<TValue>
    {
        /* UCT (upper confidence tree) winrate
         * = U_i = W_i/N_i + c * sqrt(ln(N_p)/N_i)
         * N_i visit count of child i
         * N_p visit count of parent
         * c >= 0, low c = choose lucrative nodes, else explore
        */
        public float visitCount = 0;
        public Tuple<int, int> bestMove = Tuple.Create(-1, -1);
        public int bestChildIndex = -1;
        public List<float> nn_policy = null;
        public float nn_value;
        public int moveIndex = -1; // from 0 to 8 
        public float winrate = 0.0f;
        public Node<TValue> parent = null;

        public TValue Value { get; set; }
        public List<Node<TValue>> Children { get; private set; }
        public bool HasChild { get { return Children.Any(); } }
        public Node(Node<TValue> aParent)
        {
            this.Children = new List<Node<TValue>>();
            parent = aParent;
        }
        public void AddChild(Node<TValue> treeNode)
        {
            Children.Add(treeNode);
        }
        public Node<TValue> GetParent()
        {
            return parent;
        }

    public void dirichletNoise(float epsilon, float alpha)
        {
            /*
             *     auto child_cnt = m_children.size();
                auto dirichlet_vector = std::vector<float>{};

                std::gamma_distribution<float> gamma(alpha, 1.0f);
                for (size_t i = 0; i < child_cnt; i++) {
                    dirichlet_vector.emplace_back(gamma(Random::GetRng()));
                }

                auto sample_sum = std::accumulate(begin(dirichlet_vector), end(dirichlet_vector), 0.0f);

                // If the noise vector sums to 0 or a denormal, then don't try to
                // normalize.
                if (sample_sum < std::numeric_limits<float>::min()) {
                    return;
                }

                for (auto& v: dirichlet_vector) {
                    v /= sample_sum;
                }

                child_cnt = 0;
                for (auto& child : m_children) {
                    auto winrate = child->get_score();
                    auto eta_a = dirichlet_vector[child_cnt++];
                    winrate = winrate * (1 - epsilon) + epsilon * eta_a;
                    child->set_score(winrate);
                }*/
        }
        public override string ToString()
        {
            string resultString = "Node in MCTS Tree\n\n";

            resultString += "Visit count: " + visitCount + "\n";
            resultString += "Winrate: " + winrate + "\n";
            resultString += "NN Value: " + nn_value + "\n";

            if (Value != null)
            {
                resultString += Value.ToString();
            }
            else
            {
                resultString += "No value set\n";
            }
            resultString += "\n";

            return resultString;
        }
    }
}
