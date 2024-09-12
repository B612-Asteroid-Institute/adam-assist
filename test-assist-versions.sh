#!/usr/bin/env bash

# Iterate through a list of custom dependencies and run pytest after
# each one
# e.g. "main", git+https://github.com/B612-Asteroid-Institute/assist.git@main
# set a ASSIST_VERSION environment variable with the pytest command

ASSIST_VERSIONS=("main" "ak/spk_only" "ak/spk_only-trunc40" "main-ffp-off" "ak/spk_only-ffp-off" "ak/spk_only-trunc40-ffp-off")


for version in "${ASSIST_VERSIONS[@]}"; do
    pip uninstall assist rebound -y
    pip --no-cache-dir -v install git+https://github.com/B612-Asteroid-Institute/rebound.git@$version
    pip --no-cache-dir -v install git+https://github.com/B612-Asteroid-Institute/assist.git@$version
    # replace slashes with dashes
    version=$(echo $version | tr / -)
    export ASSIST_VERSION=$version
    pytest -s -k "test_complete_residuals"
done
