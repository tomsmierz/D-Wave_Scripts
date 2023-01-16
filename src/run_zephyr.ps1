$sizes = 2, 3, 4
$categories = "AC3", "RAU", "RCO"
foreach ($size in $sizes)
{
  foreach ($cat in $categories)
  {
    $path = "C:\Users\tsmierzchalski\PycharmProjects\D-Wave_Scripts\instances\zephyr_random\Z$size\$cat"
    If(!(test-path -PathType container $path))
    {
      New-Item -ItemType Directory -Path $path
    }
    python .\generate_zephyr_instances.py -N 100 -S $size -C $cat -T SpinGlass DWave -P $path -D Advantage2_prototype1.1
  }
}
