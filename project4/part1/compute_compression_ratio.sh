# compile the KMeans program
javac *.java -d classes

# start adding header to csv files
echo "k,compression_ratio" > Koala.csv
echo "k,compression_ratio" > Penguins.csv

for n in $(seq $1); do
    for n in 2 5 10 15 20; do
        java -cp classes KMeans ./images/Koala.jpg $n ./images/Koala-$n.jpg
        python -c "print( str($n) + ',' + str($(stat -c %s ./images/Koala-$n.jpg) / $(stat -c %s ./images/Koala.jpg)) )" >> Koala.csv
    done

    for n in 2 5 10 15 20; do
        java -cp classes KMeans ./images/Penguins.jpg $n ./images/Penguins-$n.jpg
        python -c "print( str($n) + ',' + str($(stat -c %s ./images/Penguins-$n.jpg) / $(stat -c %s ./images/Penguins.jpg)) )" >> Penguins.csv
    done
done