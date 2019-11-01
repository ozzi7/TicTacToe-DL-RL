Implementation of AlphaZero (see arXiv:1712.01815 [cs.AI]) for Tic-Tac-Toe on a 5x5 board (4 in a row to win). 
The AI uses Monte Carlo Tree Search (MCTS) and a Deep Neural Network (DNN) to guide the search.
It can be trained with both backpropagation (see keras/python code) or using deep neuroevolution (as per arXiv:1703.03864 [stat.ML]).


Neural Network Structure  

Input: 5x5x3

Conv. filter size: 3x3  
Res. layers: 8  
Res. layer filters: 64  
Value filters: 64  
Policy filters: 64  

Output policy head: 25 (5x5)  
Output value head: 1 (W-L)  


MCTS Search parameters 

FPU Reduction: 0.1  
Temperature: 0.1  
Endgame temperature: 0.3  
Temperature cutoff during training: 8  
Temperature cutoff during play: 4  
FPU at root node: 1.0  


Neural network evaluations can be run on the GPU (OpenCL kernel) as well as on a CPU.
Training progress can be monitored with GnuPlot scripts.

Play vs. the AI using "TicTacToe-DL-RL.exe play weights.txt" where weights.txt contains the trained DNN weights.
