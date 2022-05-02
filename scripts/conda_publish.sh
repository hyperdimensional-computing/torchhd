#!/bin/bash

set -ex
set -o pipefail

check_if_setup_file_exists() {
    if [ ! -f setup.py ]; then
        echo "setup.py must exist in the directory that is being packaged and published."
        exit 1
    fi
}

check_if_meta_yaml_file_exists() {
    if [ ! -f meta.yaml ]; then
        echo "meta.yaml must exist in the directory that is being packaged and published."
        exit 1
    fi
}

build_package(){
    # Build for Linux
    conda build -c conda-forge -c pytorch --output-folder . .

    # Convert to other platforms: OSX, WIN
    conda convert -p osx-64 linux-64/*.tar.bz2
    conda convert -p win-64 linux-64/*.tar.bz2
}

upload_package(){
    anaconda upload --label main osx-64/*.tar.bz2
    anaconda upload --label main linux-64/*.tar.bz2
    anaconda upload --label main win-64/*.tar.bz2
}

check_if_setup_file_exists
cd ./conda  # go to build dir
check_if_meta_yaml_file_exists
build_package
upload_package