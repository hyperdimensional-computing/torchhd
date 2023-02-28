for i in 10 100 500 1000 10000
do
	for j in 1 2 3 4
	do
		/usr/bin/time -v python3 voicehd.py $i
	done
done
