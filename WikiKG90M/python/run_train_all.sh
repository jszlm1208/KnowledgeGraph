#!/bin/bash

sh ./model_dglke/train_scripts/run_ote40.sh
sh ./model_dglke/train_scripts/run_ote_hd240.sh
sh ./model_dglke/train_scripts/run_ote_lr0.3.sh
sh ./model_dglke/train_scripts/run_ote_lrd2w.sh
sh ./model_dglke/train_scripts/run_ote_lrd5k.sh
sh ./model_dglke/train_scripts/run_ote_mlplr.sh
sh ./model_dglke/train_scripts/run_ote_bs1.2k.sh
sh ./model_dglke/train_scripts/run_ote_gamma10.sh
sh ./model_dglke/train_scripts/run_ote_gamma14.sh
sh ./model_dglke/train_scripts/run_ote.sh
sh ./model_dglke/train_scripts/run_transe.sh
sh ./model_dglke/train_scripts/run_rotate.sh
sh ./model_dglke/train_scripts/run_quate.sh
