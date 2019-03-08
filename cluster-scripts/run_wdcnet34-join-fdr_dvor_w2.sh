#!/usr/bin/env bash
#module load python_gpu/3.6.4 hdf5/1.10.1
#bsub -n 1 -N -B -W 4:00 -J "test_sd" -R "rusage[mem=4096,scratch=480000, ngpus_excl_p=1]" < run_original_sparse-to-dense.sh
# Copy files to local scratch

a=v4rl_h5_dataset_v7_p1.tar
b=v4rl-ds-mini_v5.tar
ds=$a
a2=v4rl_h5_dataset_v7_p2.tar
b2=v4rl-ds-mini_v5.tar
ds2=$a2
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
python3 main.py -a weightcompnet34 -m "rgb-dvor-d2dwor" --depth-weight-head-type JOIN -s 0 -c l2gn -lr 0.01 -lrs 3 -lrm 0.00001 --depth-divider 100 --epochs 200 --pretrained $LS_SUBCWD/pretrain/visim.sparsifier=uar.samples=-1.modality=rgb-fd.arch=depthcompnet34.criterion=l2.divider=100.0.lr=0.01.lrs=5.bs=8.pretrained=True/dc34-fdr-model_best.pth.tar --data-path $TMPDIR/data/










