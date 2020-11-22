#Set current directory
CURR_DIR="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
PYENV_NAME="$(basename "$CURR_DIR")"

echo "Building python environment"

pyenv virtualenv --force 3.7.1 "pyenv.$PYENV_NAME"
pyenv local pyenv.$PYENV_NAME
pip install --upgrade pip setuptools
pip install -r requirements.txt

#Download price paid
FILE=data/pp-2020.csv

if test -f "$FILE"; then
    echo "Using existing download of price paid data"
else 
    wget http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-2020.csv -O data/pp-2020.csv
fi

echo "Running pipeline"
python run_pipeline.py