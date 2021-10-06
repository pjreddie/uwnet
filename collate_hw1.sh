rm -rf submit/
mkdir -p submit

prepare () {
    if [[ $(git diff origin -- $1 | wc -c) -eq 0 ]]; then 
        echo "WARNING: $1 is unchanged according to git."
    fi
    cp $1 submit/
}

echo "Creating tarball..."
prepare src/convolutional_layer.c
prepare src/maxpool_layer.c
prepare tryhw1.py
prepare hw1.ipynb

tar cvzf hw1.tar.gz submit
rm -rf submit/
echo "Done. Please upload hw1.tar.gz to Canvas."

