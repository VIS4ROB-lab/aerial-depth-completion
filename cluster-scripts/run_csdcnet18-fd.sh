#!/usr/bin/env bash
#module load python_gpu/3.6.4 hdf5/1.10.1
#bsub -n 1 -N -B -W 4:00 -J "test_sd" -R "rusage[mem=4096,scratch=480000, ngpus_excl_p=1]" < run_original_sparse-to-dense.sh
# Copy files to local scratch dataset_big_v8

a=dataset_big_v9.tar
b=dataset_small_v9.tar
ds=$a
a2=v4rl_h5_dataset_v7_p2.tar
b2=v4rl_h5_mini_v7_p2.tar
ds2=$b2
nvidia-smi
mkdir ${TMPDIR}/data
rsync -aq /cluster/scratch/pilucas/data/$ds ${TMPDIR}/data

# Run commands
cd $TMPDIR/data
tar -xf $ds && rm -f $ds

rsync -aq /cluster/scratch/pilucas/data/$ds2 ${TMPDIR}/data
tar -xf $ds2 && rm -f $ds2

#find
cd $LS_SUBCWD
python3 main_fusion.py -a csdepthcompnet18 -m "rgb-fd" -s 500 -c l2 -lr 0.0001 --batch-size 8 -j 16 -lrs 5 -lrm 0.000001 --depth-divider 1  --epochs 200 --data "visim_seq" --pretrained "$LS_SUBCWD/../uncertainty_aware_sparse_to_dense_seqnet/results/visim.dw_head=CBR.samples=500.modality=rgb-fd.arch=sdepthcompnet18.criterion=l2.divider=1.0.lr=0.0001.lrs=3.bs=8.pretrained=True/model_best.pth.tar" --data-path $TMPDIR/data/dataset_big_v8 

