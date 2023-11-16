# Prepare Data
# git clone https://github.com/allenai/natural-instructions.git
cp -r reproduce_splits/* natural-instructions/splits/

# Add tags into data
python3 my_scripts/write_cls_split_into_tasks.py

# Create output dir
mkdir -p ActiveIT/output/my_experiment/TLAL/

