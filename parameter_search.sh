# call example.py with different radius and friction values

for friction in 0.1 0.15 0.2 0.25 0.3 0.35 0.4; do
    for x in 0.460 0.465 0.470 0.475 0.48; do
        echo "Running with x $x and friction $friction"
        python example.py -x $x -f $friction
    done
done