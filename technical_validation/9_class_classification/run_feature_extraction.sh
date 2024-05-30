# created by DebarpanB
# date 25th August, 2022

stage=0

annotationdir='/Users/manraj/Desktop/Coswara-Data/technical_validation/9_class_classification/annotations/LABELS/'
#audiodir='/data1/srikanthr/Coswara/data_preparation/Coswara-Data-Extracted'
pathfile='technical_validation/9_class_classification/path_files/wav.scp'
audiocategory=$1

datadir_name='data'
datadir=$datadir_name/$audiocategory
feature_dir_name='feats'
feature_dir=$feature_dir_name/$audiocategory
output_dir='results'

classification_config='conf/classification.conf'
feats_config='conf/feature.conf'

/Users/manraj/Desktop/Coswara-Data/technical_validation/9_class_classification/parse_options.sh

mkdir -p $datadir
mkdir -p $feature_dir
mkdir -p $output_dir

if [ $stage -le 0 ]; then
	echo "==== Preparing lists ====="
    python3 /Users/manraj/Desktop/Coswara-Data/technical_validation/9_class_classification/local/prepare_list.py -a $annotationdir/${audiocategory}.csv -p $pathfile -l $datadir/all -s $datadir/all.scp
fi

if [ $stage -le 1 ]; then
	# Creates a separate pickle file containing feature matrix for each recording in the wav.scp
	# Expects a data folder, with train_dev and eval folders inside. each folder has a wav.scp file
	# Each row in wav.scp is formatted as: <wav_id> <wav_file_path>
	# Feature matrices are written to: $feature_dir/{train_dev/eval_set}/<wav_id>_<feature_type>.pkl
	# feature.conf specifies configuration settings for feature extraction

	echo "==== Feature extraction ====="
	mkdir -p $feature_dir
	python local/feature_extraction.py -c $feats_config -i $datadir/all.scp -o $feature_dir
	cp $feature_dir/feats.scp $datadir/feats.scp
    mv $feature_dir/bad_ids.npy $datadir/bad_ids.npy
	python local/filter_list.py -f $datadir/feats.scp -l $datadir/all -s $datadir/all.scp
fi

