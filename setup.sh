#!/bin/sh
sudo rm -R tmp/rpn_tmp/* || true
sudo rm -R experiment_save/* || true 

mkdir tmp || true
mkdir tmp/rpn_tmp || true
mkdir experiment_save || true
