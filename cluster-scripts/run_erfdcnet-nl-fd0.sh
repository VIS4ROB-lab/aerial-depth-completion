#!/usr/bin/env bash
#module load python_gpu/3.6.4 hdf5/1.10.1
#bsub -n 1 -N -B -W 4:00 -J "test_sd" -R "rusage[mem=4096,scratch=480000, ngpus_excl_p=1]" < run_original_sparse-to-dense.sh
# Copy files to local scratch

if  true 
then
	ds1=dataset-big-v10-train-2.tar
	ds2=dataset-big-v10-train-1.tar
	ds3=dataset-big-v10-train-3.tar
	ds4=dataset-big-v10-val.tar
else
	ds1=dataset-small-v10-train-2.tar
	ds2=dataset-small-v10-train-1.tar
	ds3=dataset-small-v10-train-3.tar
	ds4=dataset-small-v10-val.tar
fi

nvidia-smi
mkdir ${TMPDIR}/data
cd $TMPDIR/data

for currds in $ds1 $ds2 $ds3 $ds4
do
	echo "copying $currds..."
	rsync -aq /cluster/scratch/pilucas/data/$currds ${TMPDIR}/data
	tar -xf $currds && rm -f $currds
done

cd $LS_SUBCWD
python3 main.py -a erfdepthcompnet -m "rgb-fd" -s 0 -c l2gn -lr 0.0001 --batch-size 8 -j 16 -lrs 3 -lrm 0.000001 --depth-divider 1  --epochs 200 --pretrained resnet --data-path $TMPDIR/data/

echo
