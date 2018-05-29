#!/bin/bash

# convert from .vcf.bgz to plink data formats
chromosomes=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22")

# initialize file that will contain all files to be merged together for full data sets
rm -rf ../data/plink/merge.list
touch ../data/plink/merge.list

for chromosome in "${chromosomes[@]}"
do
  vcftools --gzvcf ../data/acb_ii_mini_project.vcf.bgz --plink --chr $chromosome --maf 0.01 --out ../data/plink/chr_$chromosome # create map/ped files
  plink --noweb --file ../data/plink/chr_$chromosome --out ../data/plink/chr_$chromosome --make-bed # convert to binary
  plink --noweb --bfile ../data/plink/chr_$chromosome --maf 0.01 --indep 50 5 1.5 --out ../data/plink/chr_$chromosome # create pruning files
  plink --noweb --bfile ../data/plink/chr_$chromosome --extract ../data/plink/chr_$chromosome.prune.in --make-bed --out ../data/plink/pruned_chr_$chromosome
  echo "../data/plink/pruned_chr_$chromosome.bed ../data/plink/pruned_chr_$chromosome.bim ../data/plink/pruned_chr_$chromosome.fam" >> ../data/plink/merge.list
done

# merge together separate chromosome files
plink --noweb --merge-list ../data/plink/merge.list --make-bed --out ../data/plink/merge

# run PCA
smartpca -i ../data/plink/merge.bed -a ../data/plink/merge.bim -b ../data/plink/merge.fam -o ../output/pca.pca -p ../output/pca.plot -e ../output/pca.eigen -l ../output/pca.log
