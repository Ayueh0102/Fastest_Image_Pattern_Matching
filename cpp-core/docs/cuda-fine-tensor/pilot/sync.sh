#!/bin/zsh
# Sends the working tree's cpp-core (without docs) and the dev scripts to C:\r-cuda\gpu5 on the lab PC.
S=/private/tmp/claude-502/-Users-Ayueh-Fastest-Image-Pattern-Matching/52012f11-5a17-4f55-9d43-acb6ee2040bd/scratchpad/gpu5
cd /Users/Ayueh/Fastest_Image_Pattern_Matching
COPYFILE_DISABLE=1 tar -cf $S/wt.tar --exclude cpp-core/docs cpp-core
setopt nullglob; cd $S && COPYFILE_DISABLE=1 tar -rf wt.tar *.bat *.ps1 *.py
/Applications/Tailscale.app/Contents/MacOS/Tailscale file cp $S/wt.tar ptri-pro-4000:
