filepath = ".\\bin\\Release\\plotdata.txt"
set terminal wxt size 1280,800

while (1) {
	set key left top
	set key box
	set multiplot

	set size 1.0,0.5
	set origin 0,0.5
	set yrange [0:1]
	set xtics 50
	set ytics 0.1
	set y2tics 0.1
	set grid ytics

	plot filepath using 2 with lines title "Winrate Player X (starts) [against self]" lw 3,\
	'' using 3 with lines title "Winrate Player Z [against self]" lw 3,\
	'' using 4 with lines title "Drawrate [against self]" lw 3,\
	'' using 6 with lines title "Winrate [against random]" lw 3

	set ytics auto
	set y2tics auto
	set xtics auto
	set size 0.5,0.5
	set origin 0,0.0
	set key left top
	unset yrange
	plot filepath using 1 with lines title "Self-play ELO" lw 3


	set size 0.5,0.5
	set origin 0.5,0.0
	set key right top
	unset yrange
	plot filepath using 5 with lines title "Avg. Game Length" lw 3
	
	pause 2
}
