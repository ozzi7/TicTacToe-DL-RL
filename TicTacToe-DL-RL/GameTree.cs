using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    public class GameTree<TValue>
    {
        public TValue Value { get; set; }
        public List<GameTree<TValue>> Children { get; private set; }
        public bool HasChild { get { return Children.Any(); } }
        public GameTree()
        {
            this.Children = new List<GameTree<TValue>>();
        }
        public GameTree(TValue value)
            : this()
        {
            this.Value = value;
        }
        public void AddChild(GameTree<TValue> treeNode)
        {
            Children.Add(treeNode);
        }
        public void AddChild(TValue value)
        {
            var treeNode = new GameTree<TValue>(value);
            AddChild(treeNode);
        }
    }
}
