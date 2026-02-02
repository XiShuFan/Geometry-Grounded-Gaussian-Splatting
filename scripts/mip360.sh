dataset_folder=/media/baowen/data/dataset/360_v2
for scene in room counter kitchen bonsai; do
    python train.py -s ${dataset_folder}/${scene} -m output/mip360_sg/${scene} -r 2 --sh_degree 2 --sg_degree 7 --eval
    python render.py -m output/mip360_sg/${scene}
    python metric.py -m output/mip360_sg/${scene}
done

for scene in bicycle garden stump flowers treehill; do
    python train.py -s ${dataset_folder}/${scene} -m output/mip360_sg/${scene} -r 4 --sh_degree 2 --sg_degree 7 --eval
    python render.py -m output/mip360_sg/${scene}
    python metric.py -m output/mip360_sg/${scene}
done


for scene in room counter kitchen bonsai; do
    python train.py -s ${dataset_folder}/${scene} -m output/mip360/${scene} -r 2 --eval
    python render.py -m output/mip360/${scene}
    python metric.py -m output/mip360/${scene}
done

for scene in bicycle garden stump flowers treehill; do
    python train.py -s ${dataset_folder}/${scene} -m output/mip360/${scene} -r 4 --eval
    python render.py -m output/mip360/${scene}
    python metric.py -m output/mip360/${scene}
done
