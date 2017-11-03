#!/usr/bin/env zsh
set -e
mkdir -p examples/mnist/log
LOG=examples/mnist/log/log-$(date +%Y-%m-%d-%H-%M-%S).log
./build/tools/caffe train --solver=examples/mnist/lenet_solver.prototxt 2>&1 | tee $LOG $@
