i=0
max=10
echo "usage: sh convert_all_textures.sh base_path object"
while [ $i -lt $max ]
do
    echo ">>>> Texture $i of object ${2} <<<"
    python3 ./tod_converter_stereo.py -d ${1}/${2}_orig/ -m ${1}/metafiles/${2} -o ${1}/sy_datasets/${2}_stereo_$i/ -t $i -c ${2}
    true $((i=i+1))
done