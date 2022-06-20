path1=$2
path2=$3
python session.py -mode test\
 -cfg DS_path=$path1 US_path=$path2\
 debugging=False\
 beam_search=False beam_size=10\
 DS_device=$1 US_device=$1