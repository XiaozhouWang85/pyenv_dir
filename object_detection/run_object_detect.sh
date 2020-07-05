#Set current directory
export CURR_DIR="$(dirname "${BASH_SOURCE[0]}")"
export PARENT_DIR="$(cd $CURR_DIR; cd ../ && pwd)"

#Execute init.sh in parent directory to build pyenv
source $PARENT_DIR/init.sh

#Move to script location and run script
cd $PARENT_DIR/$CURR_DIR

#Download YOLO model
FILE=yolov3.weights

if test -f "$FILE"; then
    echo "Using existing download of yolov3.weights"
else 
    wget https://pjreddie.com/media/files/yolov3.weights
fi

#Prepare directories
rm -rf checkpoints
mkdir checkpoints

#Run python scripts
python run_object_detect.py 

