using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TicTacToe_DL_RL
{
    public static class ID
    {
        public static int current_ID = 0;
        public static int GetGlobalID()
        {
            return current_ID++;
        }
        public static void ResetGlobalID()
        {
            current_ID = 0;
        }
    }
}
