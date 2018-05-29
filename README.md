# MacArthur Lab ACB II Mini Project
## Christian Covington

<hr>

# Table of Contents
1. [Setup and Necessary Tools](#Setup)
2. [Data Choices](#Data-Choices)
3. [Analytic Choices](#Analytic-Choices)
4. [Code](#Code)
5. [Output](#Output)

<hr>

### Setup and Necessary Tools <a name='Setup'></a>

As recommended in the project description, I downloaded the following packages. I've included steps that worked for me on Ubuntu Linux.
1. Install [vcftools](#https://vcftools.github.io/index.html). In order to use `vcftools` from the command line, you will likely need to add the directory location to your path. For example, I installed `vcftools` in `/home/christian/applications/` and I added the following to my .bashrc file:
```
export PATH="/home/christian/applications/vcftools_0.1.13/bin:$PATH"
```
2. Install [PLINK](#http://zzz.bwh.harvard.edu/plink/). Getting this to work varies a bit by OS, but on Linux it is as easy as copying the executable from the `.zip` file to a location on your path. I chose to copy it to `usr/local/bin`

3. Install [eigensoft](https://data.broadinstitute.org/alkesgroup/EIGENSOFT/EIG-6.1.4.tar.gz). You can follow very similar instructions to `vcftools`. I ended up installing to the same `applications` directory and adding the following to my .bashrc file
```
export PATH="/home/christian/applications/EIG-6.1.4/bin:$PATH"
```
Note that I am using version 6.1.4 (rather than the newest version 7.2.1) because I could not get 7.2.1 to run on my computer. I also commented out the `ploteig` portion of the `smartpca` code because `smartpca` couldn't find `smarteig`.

<hr>

### Data Choices <a name='Data-Choices'></a>
- Filtering
  - Alleles dropped if minor allele frequency < 0.01 -- done via `vcftools`
  - SNPs dropped if VIF is greater than 1.5 (for each SNP, we look around the SNP with a window size of 50 and shift the window by 5 each step) -- done via `PLINK`
    - The parameters above (VIF threshold, window size, and window shift) were the values recommended in the [PLINK documentation](#http://zzz.bwh.harvard.edu/plink/summary.shtml). If I had more content knowledge/experience, I might have chosen different values.

<hr>

### Analytic Choices <a name='Analytic-Choices'></a>
- PCA
  - I used the standard options given by `eigensoft` to perform the PCA. This normalizes genotypes automatically and reports the top 10 principal components.
- Ancestry Classification
  - I used k-Nearest Neighbors (KNN) with k = 5 as my classification scheme. I used all 10 principal components generated from the eigensoft smartpca in the model. I chose to use all 10 principal components in the model simply because it was the number given by `eigensoft` and seemed reasonable (not a large enough number to cause problems for KNN -- more on this later). Perhaps a better way to do this would be to note how much additional variance each principal component explains in our data and choose which to include in the model based on this.
    - Given my lack of experience with this type of data, I wanted a method that was relatively simple and robust. I consider KNN to fit both of these criteria, so I then considered under which circumstances KNN is not a good method. I think of KNN having two major problems; it can be slow to train on very large datasets and it tends to do poorly in very high-dimensional settings. Our data were relatively small, so the runtime was not an issue. By running the PCA we reduced the dimension of our data to a very manageable size, so the 'curse of dimensionality' was also not an issue.
    - Because KNN fit all my criteria, I didn't really consider other options. If pressed, I think the next method I would try would be the naive Bayes classifier. Some people don't like using naive Bayes because it relies on strong independence assumptions (independence of features conditional on the class), but they often work very well in practice.
  - I split the individuals with ancestry information into training (80%) and test (20%) sets. I trained the model on the training set and used the test set to estimate out-of-sample performance. On the test set, the model classified ~97% of individuals correctly. I then applied the model to the observations with missing ancestry information.
  - The choice of k and the train/test split were both done because they were standard options.

<hr>

### Code <a name='Code'></a>

- `00_raw_to_pca.sh`
  - Unzip raw data, filter data, convert to PLINK format, and run PCA
- `01_pca_visualization_modeling.py`
  - Load PCA data, predict ancestry, and visualize PCA vs. ancestry relationship in various subsets

<hr>

### Output <a name='Output'></a>

- `pca.*`
  - all `pca.` files are output from the eigensoft `smartpca` -- generated in `00_raw_to_pca.sh`
- `pca/`: directory of output from `01_pca_visualization_modeling.py`
  - `full_pca_df.csv`
    - full data with IDs, sample (test, train, missing ancestry), observed and predicted ancestry, and PCA values
  - `test_set_confusion_matrix.csv`
    - confusion matrix showing predicted vs. observed ancestries for individuals in our test set
  - `observed_pca.png`:
    - scatterplot showing observed ancestry across top two principal components -- does not include individuals with missing ancestry
  - `predicted_pca_full.png`
    - scatterplot showing predicted ancestry across top two principal components -- includes entire sample (train, test, and missing)
  - `predicted_pca_missing.png`
    - scatterplot showing predicted ancestry across top two principal components -- includes only individuals with missing ancestry
  - `predicted_pca_test.png`:
    - scatterplot showing predicted ancestry across top two principal components -- includes only individuals in test set
