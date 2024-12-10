#!/bin/bash
# set -x
export CUDA_VISIBLE_DEVICES=0
gs_iteration=20000 # 95001 减去GS的预训练iterations为实际训练iterations
train_port=6000

# ---------------------------------dir conf---------------------------------
dir_base=/home/xhd/xhd/GaussianRoom
dir_dataset=/home/xhd/xhd/0-dataset/GaussianRoomData
dir_sdf_output=/home/xhd/xhd/0-output/test009/sdf_output
dir_gs_output=/home/xhd/xhd/0-output/test009/gs_output

# ---------------------------------exp conf---------------------------------
exp="1234567"
pno_sam_iter=2500
pgeo_interval=100
psam_add_len=2.0
panchor_thres=0.01
panchor_interval=999
panchor_knn=10
panchor_extend=0.0156
psdf2gs_from=6_000
psdf2gs_end=60_000
pgs2sdf_from=6_000
pgs_post_iter=100_000

# for scene in 0050_00 0085_00 0114_02 0580_00 0603_00 0616_00 0617_00 0721_00
for scene in 0085_00 
do
    # for exp_name in full no_norm no_edge no_geo no_sam no_anchor
    for exp_name in full
    do
        # -----------------------------train-----------------------------

        if [ "$exp_name" == "full" ]; then
            exp="1234567"
        fi

        if [ "$exp_name" == "no_norm" ]; then
            exp="0034567"
        fi

        if [ "$exp_name" == "no_edge" ]; then
            exp="1200567"
        fi

        if [ "$exp_name" == "no_geo" ]; then
            exp="1234067"
        fi

        if [ "$exp_name" == "no_sam" ]; then
            exp="1234507"
        fi

        if [ "$exp_name" == "no_anchor" ]; then
            exp="1234560"
        fi

        
        python train.py \
            -s ${dir_dataset}/${scene}-GS \
            -m ${dir_gs_output} \
            --eval \
            --iterations ${gs_iteration} \
            --port ${train_port} \
            --start_checkpoint ${dir_dataset}/pretrained_${scene}/chkpnt15000.pth \
            --mode train \
            --conf ${dir_base}/confs/template.conf \
            --scene_name scene${scene} \
            --exp_name scene${scene}-${exp_name} \
            --data_dir ${dir_dataset} \
            --exp_dir ${dir_sdf_output} \
            --anchor_interval ${panchor_interval} \
            --anchor_extend ${panchor_extend} \
            --anchor_knn ${panchor_knn} \
            --no_sam_iter ${pno_sam_iter} \
            --geo_interval ${pgeo_interval} \
            --anchor_thres ${panchor_thres} \
            --sdf2gs_from ${psdf2gs_from}\
            --sdf2gs_end ${psdf2gs_end}\
            --gs2sdf_from ${pgs2sdf_from}\
            --gs_post_iter ${pgs_post_iter}\
            --sam_add_len ${psam_add_len} \
            --exp_conf ${exp} 
            

        # -----------------------------val mesh-----------------------------
        neuris_exp_path=${dir_sdf_output}/neus/scene${scene}/scene${scene}-${exp_name}
        path_conf=${neuris_exp_path}/scene${scene}-${exp_name}.conf
        dir_results_baseline=${neuris_exp_path}/meshes
        # path_mesh_pred=${dir_results_baseline}/00010000_reso512_scene0085_00_world.ply

        python ./NeuRIS-eval/exp_runner.py \
            --mode validate_mesh \
            --conf $path_conf \
            --is_continue
        
        # path_mesh_pred=$(find "${dir_results_baseline}" -name "world")
        path_mesh_pred=$(find "${dir_results_baseline}" -type f -name "*world.ply")
        python ./NeuRIS-eval/exp_evaluation.py \
            --mode eval_3D_mesh_metrics \
            --dir_dataset $dir_dataset \
            --dir_results_baseline $dir_results_baseline \
            --path_mesh_pred $path_mesh_pred \
            --scene_name scene${scene}
        
        # -----------------------------val render-----------------------------
        dir_output=${dir_gs_output}/scene${scene}-${exp_name}
        python render.py -m $dir_output
        python metrics.py -m $dir_output

    done 

done