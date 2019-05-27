set grid
set terminal wxt 0
plot ".\\bin\\Release\\plotdata.txt" using 2 with lines title "Winrate Player X (starts) [against self]" lw 3,\
'' using 3 with lines title "Winrate Player Z [against self]" lw 3,\
'' using 4 with lines title "Drawrate [against self]" lw 3,\
'' using 6 with lines title "Winrate [against random]" lw 3

set terminal wxt 1
plot ".\\bin\\Release\\plotdata.txt" using 1 with lines title "Pseudo ELO" lw 3

set terminal wxt 2
plot ".\\bin\\Release\\plotdata.txt" using 5 with lines title "Avg. Game Length" lw 3
pause 2
reread