filepath = ".\\TicTacToe-DL-RL\\bin\\Release\\plotdata.txt"
set terminal wxt size 600,600

while (1) {
	set key opaque
	set key left top
	set key box
	set multiplot

	set size 1.0,0.5
	set origin 0,0.5
	set grid
	plot filepath using 10 with lines title "Training X wins/games [against self]" lw 3,\
	'' using 11 with lines title "Training Z wins/games [against self]" lw 3,\
	'' using 13 with lines title "Start board value" lw 3
	
	set size 1.0,0.5
	set origin 0,0
	set grid
	plot filepath using 12 with lines title "Training average moves [against self]" lw 3, \
	
	pause 2
}
