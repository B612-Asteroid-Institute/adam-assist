#!/usr/bin/env bash

# Iterate through a list of custom dependencies and run pytest after each one
# Set an ASSIST_VERSION environment variable with the pytest command

ASSIST_BRANCH=("main" "ak/spk_only")
FFP=("off" "on")

for branch in "${ASSIST_BRANCH[@]}"; do
    for ffp in "${FFP[@]}"; do
        if [ "$ffp" == "on" ]; then
            # delete environment variable
            unset FFP_CONTRACT_OFF
        else
            export FFP_CONTRACT_OFF=1
        fi
        # Only the 'ak/spk_only' branch has truncation options
        if [ "$branch" == "main" ]; then
            TRUNCATION=("0")
        else
            TRUNCATION=("0" "50")
        fi
        for truncation in "${TRUNCATION[@]}"; do
            pip uninstall assist rebound -y
            echo "Running test with version $branch, truncation $truncation, and ffp $ffp"

            # Remove spaces around '=' in variable assignments
            SAFE_BRANCH=$(echo "$branch" | tr / -)
            export ASSIST_VERSION="$SAFE_BRANCH-trunc-$truncation-ffp-$ffp"
            export ASSIST_TRUNCATE="$truncation"

            # Use the pip_install function to handle optional compiler flags
            pip install --no-cache-dir -v git+https://github.com/B612-Asteroid-Institute/rebound.git@main
            pip install --no-cache-dir -v git+https://github.com/B612-Asteroid-Institute/assist.git@$branch

            # Export environment variables

            # Run pytest
            pytest -s -k "test_horizons_residuals"
        done
    done
done