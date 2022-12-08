$sizes = 2, 3, 4
$categories = "RAU", "RCO"
foreach ($size in $sizes)
{
  foreach ($cat in $categories)
  {

    python generate_zephyr_instances.py -N 100 -S $size -C $cat -T SpinGlass DWave -P "C:\Users\tsmierzchalski\PycharmProjects\D-Wave_Scripts\instances\zephyr_random\Z$size\$cat" -D Advantage2_prototype1.1
  }
}
