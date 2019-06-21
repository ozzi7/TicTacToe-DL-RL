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
        public float visits = 0;
        public List<float> nn_policy = null;
        public float nn_value;
        public int moveIndex = -1; // from 0 to 24 
        public float scoreSum = 0.0f; // from -inf to inf, where high values = X is winning
        public float winrate = 0.0f;
        public int virtualVisits = 0; // a visit which needs a NN eval increases this
        public bool waitingForGPUPrediction = false; // if this node is already waiting for a GPU eval

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
        public override string ToString()
        {
            string resultString = "Node in MCTS Tree\n\n";

            resultString += "Visit count: " + visits + "\n";
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
