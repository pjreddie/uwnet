rm -rf submit/
mkdir -p submit

prepare () {
    if [[ $(git diff origin -- $1 | wc -c) -eq 0 ]]; then 
        echo "WARNING: $1 is unchanged according to git."
    fi
    cp $1 submit/
}

echo "Creating tarball..."
prepare src/batchnorm_layer.c
prepare tryhw2.py
prepare hw2.ipynb

tar cvzf hw2.tar.gz submit
rm -rf submit/
echo "Done. Please upload hw2.tar.gz to Canvas."

