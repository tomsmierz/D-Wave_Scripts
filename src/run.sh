#!/bin/sh
for size in 4 8 12 16
do
  for cat in RAU RCO AC3
  do
    python generate_pegasus_instances.py -N 100 -S $size -C $cat -T SpinGlass DWave -P "C:\Users\tsmierzchalski\PycharmProjects\D-Wave_Scripts\instances\pegasus_random\P$size\$cat" -D Advantage_system6.1
  done
done