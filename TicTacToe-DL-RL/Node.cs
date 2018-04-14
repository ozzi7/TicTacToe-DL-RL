/*
 * Used for MCTS
<<<<<<< HEAD
 * Nodes have a board position and some metadata such as upper confidence score
=======
>>>>>>> e402893c6ab723b8426140b7615569a223867169
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
        /* UCT (upper confidence tree) score
         * = U_i = W_i/N_i + c * sqrt(ln(N_p)/N_i)
         * N_i visit count of child i
         * N_p visit count of parent
         * c >= 0, low c = choose lucrative nodes, else explore
        */
<<<<<<< HEAD
=======
        public double UCT_score_initial = 0.0;
        public double UCT_score = 0.0;
        public bool visited = false;
        public int N_p = 0;
>>>>>>> e402893c6ab723b8426140b7615569a223867169

        public TValue Value { get; set; }
        public List<Node<TValue>> Children { get; private set; }
        public bool HasChild { get { return Children.Any(); } }
        public Node()
        {
            this.Children = new List<Node<TValue>>();
        }
        public Node(TValue value)
            : this()
        {
            this.Value = value;
        }
        public void AddChild(Node<TValue> treeNode)
        {
            Children.Add(treeNode);
        }
        public void AddChild(TValue value)
        {
            var treeNode = new Node<TValue>(value);
            AddChild(treeNode);
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
        auto score = child->get_score();
        auto eta_a = dirichlet_vector[child_cnt++];
        score = score * (1 - epsilon) + epsilon * eta_a;
        child->set_score(score);
    }*/
        }
    }
}
