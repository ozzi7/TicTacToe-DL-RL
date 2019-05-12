set grid
set terminal wxt 0
plot ".\\bin\\Release\\plotdata.txt" using 2 with lines title "Winrate Player X (starts)",\
'' using 3 with lines title "Winrate Player Z",\
'' using 4 with lines title "Drawrate"

set terminal wxt 1
plot ".\\bin\\Release\\plotdata.txt" using 1 with lines title "Pseudo ELO"

set terminal wxt 2
plot ".\\bin\\Release\\plotdata.txt" using 5 with lines title "Avg. Game Length"
pause 20
reread