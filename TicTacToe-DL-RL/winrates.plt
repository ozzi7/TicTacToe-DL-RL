set grid
set multiplot layout 1,4
set title "1" 
plot ".\\bin\\Release\\plotdata.txt" using 2 with lines title "Winrate Player X (starts) [against self]",\
'' using 3 with lines title "Winrate Player Z [against self]",\
'' using 4 with lines title "Drawrate [against self]",\
'' using 6 with lines title "Winrate [against random]"

set title "2" 
plot ".\\bin\\Release2\\plotdata.txt" using 2 with lines title "Winrate Player X (starts) [against self]",\
'' using 3 with lines title "Winrate Player Z [against self]",\
'' using 4 with lines title "Drawrate [against self]",\
'' using 6 with lines title "Winrate [against random]"

set title "3" 
plot ".\\bin\\Release3\\plotdata.txt" using 2 with lines title "Winrate Player X (starts) [against self]",\
'' using 3 with lines title "Winrate Player Z [against self]",\
'' using 4 with lines title "Drawrate [against self]",\
'' using 6 with lines title "Winrate [against random]"

set title "4" 
plot ".\\bin\\Release4\\plotdata.txt" using 2 with lines title "Winrate Player X (starts) [against self]",\
'' using 3 with lines title "Winrate Player Z [against self]",\
'' using 4 with lines title "Drawrate [against self]",\
'' using 6 with lines title "Winrate [against random]"

pause 20
reread