import os
import random
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

'''
1. Create PCA dataframe
2. Use PCA to predict ancestry for individuals missing ancestry data
3. Visualize PCA/ancestry relationship
'''
def main():
    random.seed(1)
    # set output path
    pca_path = '../output/pca/'
    if not os.path.exists(pca_path):
        os.makedirs(pca_path)

    # store PCA data frame and array of columns corresponding to PCA values
    df, pca_cols = createPCAData()

    # use principal components to predict ancestry
    predictAncestry(df, pca_cols,
                    confusion_path = '%stest_set_confusion_matrix.csv' % pca_path,
                    predictions_path = '%smissing_predicted_ancestries.csv' % pca_path,
                    full_df_path = '%sfull_pca_df.csv' % pca_path)

    # create visualization of first two principal components
    visualizePCA(df, pca_path = pca_path)

def createPCAData():
    '''
    Set up PCA data frame
    '''
    # read in ID/ancestry data
    id_anc = pd.read_csv('../data/acb_ii_mini_project_labels.txt', delimiter = '\t')
    id_anc.columns = ['ID', 'ancestry']

    # read in PCA data
    # NOTE: this can't be the best way to do this, but I couldn't get the built-in pandas
    # methods to work properly
    df = pd.DataFrame(index = np.arange(0,id_anc.shape[0]),
                      columns=['ID'] + ['PCA%s' % x for x in range(1,11)] + ['Missing_Col'])
    pca_array = []
    with open('../output/pca.pca.evec', 'r') as f:
        for line in f:
            pca_array.append(line)

    pca_array = np.array(pca_array)
    for i in np.arange(1, pca_array.shape[0]):
        row = pca_array[i].split()
        df.loc[i - 1] = row

    # clean up PCA data frame
    del df['Missing_Col'] # delete unnecessary column
    df['ID'] = df['ID'].str.split(':').apply(pd.Series)[0] # clean up ID name in main PCA data
    df = df.merge(right = id_anc, how = 'left', on = 'ID') # merge ancestry onto main PCA data

    # change data types of columns as necessary
    pca_cols = ['PCA%s' % x for x in range(1,11)] # identify columns associated with principal components
    df['ancestry'] = df['ancestry'].astype('category') # change ancestry to category variable
    df[pca_cols] = df[pca_cols].apply(pd.to_numeric, errors='coerce', axis = 1) # change principal component columns to numeric

    return(df, pca_cols)

def predictAncestry(df, pca_cols, confusion_path, predictions_path, full_df_path):
    '''
    Predict ancestry for observations with missing ancestry
    '''
    # create column noting sample for each observation (train, test, or missing ancestry)
    df['selection_num'] = np.random.uniform(low = 0, high = 1, size = df.shape[0]) # random number that will serve as basis for train/test split
    conditions = [(pd.notnull(df['ancestry'])) & (df['selection_num'] < 0.8),
                  (pd.notnull(df['ancestry'])) & (df['selection_num'] >= 0.8),
                  (pd.isnull(df['ancestry']))]
    choices = ['train', 'test', 'missing']
    df['sample'] = np.select(conditions, choices) #
    del df['selection_num'] # delete random number column, as it is now extraneous

    # separate data into observations with/without ancestry -- split observations with ancestry into train/test sets
    train = df.loc[(df['sample'] == 'train')]
    test = df.loc[(df['sample'] == 'test')]
    missing_anc_df = df.loc[(df['sample'] == 'missing')]

    # fit nearest-neighbor model
    classifier = KNeighborsClassifier(n_neighbors = 5) # define knn model
    classifier.fit(train[pca_cols], train['ancestry']) # fit knn model
    test_pred = classifier.predict(test[pca_cols]) # predict ancestry

    '''
    Check test-set performance and save to csv
    '''
    # test set performance
    test_performance = len([i for i, j in zip(test_pred, test['ancestry']) if i == j]) / len(test_pred) # proportion of predictions that are correct
    test_actual = pd.Series(pd.Categorical(test['ancestry'], categories = ['afr', 'amr', 'eas', 'fin', 'nfe', 'sas']), name = 'Actual/Predicted') # actual ancestries for test set
    test_pred = pd.Series(pd.Categorical(test_pred, categories = ['afr', 'amr', 'eas', 'fin', 'nfe', 'sas'])) # predicted ancestries for test set

    # create confusion matrix
    df_confusion = pd.crosstab(test_actual, test_pred) # create crosstab
    df_confusion = df_confusion.reindex(['afr', 'amr', 'eas', 'fin', 'nfe', 'sas'], axis = 0) # fix labels
    df_confusion.to_csv(confusion_path) # save confusion matrix to csv
    with open(confusion_path, 'a') as f:
        f.write('\ntest set performance: ' + str(test_performance) + ' proportion correct')

    # get predictions for full pca df
    full_pred = classifier.predict(df[pca_cols]) # predict ancestry on full data
    df.loc[:, ('pred_ancestry')] = pd.Categorical(list(full_pred)) # change predicted ancestry to categorical
    reorder_cols = ['ID', 'sample', 'ancestry', 'pred_ancestry'] + pca_cols # reorder columns for better presentation
    df = df[reorder_cols]
    df.to_csv(full_df_path)

def visualizePCA(df, pca_path):
    '''
    Visualize PCA
    '''
    # set dot size for all PCA plots
    dotsize = 5

    # visualize PCA for observed (actual) ancestries
    pca_plt = seaborn.pairplot(x_vars=['PCA1'], y_vars=['PCA2'], data = df.loc[(df['sample'] != 'missing')], hue='ancestry', size = dotsize)
    pca_plt.savefig(pca_path + 'observed_pca.png')

    # visualize PCA for predicted ancestries for full sample
    pca_plt = seaborn.pairplot(x_vars=['PCA1'], y_vars=['PCA2'], data = df, hue='pred_ancestry', size = dotsize)
    pca_plt.savefig(pca_path + 'predicted_pca_full.png')

    # visualize PCA for predicted ancestries for test set
    pca_plt = seaborn.pairplot(x_vars=['PCA1'], y_vars=['PCA2'], data = df.loc[(df['sample'] == 'test')], hue='pred_ancestry', size = dotsize)
    pca_plt.savefig(pca_path + 'predicted_pca_test.png')

    # visualize PCA for predicted ancestries for missing data
    pca_plt = seaborn.pairplot(x_vars=['PCA1'], y_vars=['PCA2'], data = df.loc[(df['sample'] == 'missing')], hue='pred_ancestry', size = dotsize)
    pca_plt.savefig(pca_path + 'predicted_pca_missing.png')

# run function
if __name__ == '__main__':
    main()
