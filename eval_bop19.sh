result=$1
bop19_json=$2
csv_path=$3
bop_json=${4:-'test_targets_bop19.json'}
python tools/convert_to_bop19.py $result $bop19_json $csv_path
cp $csv_path ~/packs/bop_toolkit/results/

cd ~/packs/bop_toolkit
Xvfb :22 -screen 0 1024x768x16 &
export DISPLAY=:22
python scripts/eval_bop19_pose.py --renderer_type=vispy --result_filenames ${csv_path##*/} --targets_filename ${bop_json}