filepath = ".\\TicTacToe-DL-RL\\bin\\Release\\plotdata.txt"
set terminal wxt size 800,800

while (1) {
	set key opaque
	set key left bottom
	set key box
	set multiplot

	set size 1.0,0.5
	set origin 0,0.5
	set ytics 0.1
	set y2tics 0.1
	set grid

	plot filepath using 2 with lines title "Winrate Player X (starts) [against self]" lw 3,\
	'' using 3 with lines title "Winrate Player Z [against self]" lw 3,\
	'' using 4 with lines title "Drawrate [against self]" lw 3,\
	'' using 6 with lines title "Winrate [against random 80 nodes]" lw 3, \
	'' using 7 with lines title "Winrate [against random 10 nodes]" lw 3, \
	'' using 8 with lines title "Winrate [against random 1 node]" lw 3 linetype 14

	set ytics auto
	set y2tics auto
	set size 0.5,0.5
	set origin 0,0
	set key left bottom
	unset yrange
	plot filepath using 1 with lines title "Self-play ELO" lw 3,\
	'' using 9 with lines title "Self-play ELO incl. discarded" lw 3

	set size 0.5,0.5
	set origin 0.5,0
	set key right bottom
	unset yrange
	plot filepath using 5 with lines title "Avg. Game Length" lw 3
	
	
	pause 2
}
