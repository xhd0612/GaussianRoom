export CUDA_VISIBLE_DEVICES=0
gs_iteration=95001

# --------------------------------------------------------------------------------------------
dir_base=/home/xhd/xhd/3DGS_SDF
dir_dataset=/home/xhd/xhd/0-dataset/neuris_data
dir_sdf_output=/home/xhd/xhd/0-output/neuris_data_sdf
dir_gs_output=/home/xhd/xhd/0-output/neuris_data_gs

# for scene in 0050_00 0085_00 0114_02 0580_00 0603_00 0616_00 0617_00 0721_00
for scene in 0050_00
do
    # branch 为True
    # for exp_name in full no_norm no_edge no_geo_gui no_sample_gui no_anchor
    for exp_name in full no_norm no_edge no_geo_gui no_sample_gui no_anchor
    do
        # -----------------------------train-----------------------------
        exp="1234567"
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