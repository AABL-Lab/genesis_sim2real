# call example.py with different radius and friction values

for friction in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    for radius in 0.0275 0.03 0.0325 0.035; do
        echo "Running with radius $radius and friction $friction"
        python example.py -r $radius -f $friction
    done
done