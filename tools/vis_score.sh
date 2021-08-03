for i in $(seq 0.3 0.05 0.95);
do
	python3 test_r3det.py --vis_score $i;
done
