#!/bin/sh
sudo rm -R tmp/rpn_tmp || true
sudo rm -R experiment_save/* || true 

mkdir tmp
mkdir tmp/rpn_tmp
mkdir experiment_save
