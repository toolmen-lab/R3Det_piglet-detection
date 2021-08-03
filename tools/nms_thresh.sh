for i in $(seq 0 0.05 0.5);
do
	python3 test_r3det.py --nms_thresh $i;
done
