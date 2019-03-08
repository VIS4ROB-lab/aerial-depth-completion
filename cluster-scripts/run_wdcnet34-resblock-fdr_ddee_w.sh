#!/usr/bin/env bash
#module load python_gpu/3.6.4 hdf5/1.10.1
#bsub -n 1 -N -B -W 4:00 -J "test_sd" -R "rusage[mem=4096,scratch=480000, ngpus_excl_p=1]" < run_original_sparse-to-dense.sh
# Copy files to local scratch

a=v4rl_h5_dataset_v6.tar
b=v4rl-ds-mini_v5.tar
ds=$a
nvidia-smi
mkdir ${TMPDIR}/data
rsync -aq /cluster/scratch/pilucas/data/$ds ${TMPDIR}/data

# Run commands
cd $TMPDIR/data
tar -xf $ds && rm -f $ds
#find
cd $LS_SUBCWD
python3 main.py -a weightcompnet34 -m "rgb-ddee-d3dwor" --depth-weight-head-type ResBlock1 -s 0 -c l2 -lr 0.01 -lrs 3 -lrm 0.00001 --depth-divider 100 --epochs 200 --pretrained $LS_SUBCWD/pretrain/visim.sparsifier=uar.samples=-1.modality=rgb-fd.arch=depthcompnet34.criterion=l2.divider=100.0.lr=0.01.lrs=5.bs=8.pretrained=True/dc34-fdr-model_best.pth.tar --data-path $TMPDIR/data/










