filepath = ".//Training//training_loss.txt"

set terminal wxt size 800,500

while (1) {
	set key opaque
	set key right top
	set key box

	set grid ytics

	plot filepath matrix every 1::0 with lines title "training loss"
	
	pause 3
}
