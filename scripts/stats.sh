#!/bin/bash

# ==================================================================================================
# Project: 
# Exploring the performance of general matrix-multiplication on an NVIDIA Tesla K40m GPU.
#
# File information:
# Institution.... SURFsara <www.surfsara.nl>
# Author......... Cedric Nugteren <cedric.nugteren@surfsara.nl>
# Changed at..... 2014-10-30
# License........ MIT license
# Tab-size....... 4 spaces
# Line length.... 100 characters
#
# ==================================================================================================

# Read the filename from the command-line
file=$1

# Calculate occurences of particular instructions in the assembly
FFMA=`cat $file | grep -c "FFMA"`
LDS=`cat $file | grep -c "LDS"`
STS=`cat $file | grep -c "STS"`
SHFL=`cat $file | grep -c "SHFL"`
LD=`cat $file | grep -c "LD[^S]"`
ST=`cat $file | grep -c "ST[^S]"`
MOV=`cat $file | grep -c "MOV"`
SUM=$((FFMA+LDS+STS+SHFL+LD+ST+MOV+SUM))

# Print the resulting statistics to screen
echo ">> Stats on $file:"
echo ">> "
echo ">> FFMA  $FFMA"
echo ">> LDS   $LDS"
echo ">> STS   $STS"
echo ">> SHFL  $SHFL"
echo ">> LD    $LD"
echo ">> ST    $ST"
echo ">> MOV   $MOV"
echo ">> "
echo ">> TOTAL=$SUM"

# ==================================================================================================
