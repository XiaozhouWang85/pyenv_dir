#Set current directory
export CURR_DIR="$(dirname "${BASH_SOURCE[0]}")"
export PARENT_DIR="$(cd $CURR_DIR; cd ../ && pwd)"

#Execute init.sh in parent directory to build pyenv
source $PARENT_DIR/init.sh

#Move to script location and run script
cd $PARENT_DIR/$CURR_DIR
#python run_lsh.py
python download_sift.py 