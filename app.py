#It takes time to run the program, please be patient after opening the url
from shiny import ui, render, App
import numpy as np
import pandas as pd
from matplotlib.pyplot import subplots
import statsmodels.api as sm
from ISLP import load_data
from ISLP.models import (ModelSpec as MS,summarize)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, auc, roc_curve,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

from ISLP import confusion_table
from ISLP.models import contrast
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import time
import datetime
import re

#load our data
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

features = data.data
target = data.target
df = pd.DataFrame(features, columns=data.feature_names)
df['target'] = target

#down-sample the dataset
#we have 1 more than 0, better down-sample the 1 part
n_t1 = 500 - (df["target"].count() - df["target"].sum())
df_t1 = df[df['target'] == 1]
df_t0 = df[df['target'] == 0]
df_t1_sampled = df_t1.sample(n=n_t1, random_state=42)
df_final = pd.concat([df_t1_sampled, df_t0])
df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)

#compute the correlation matrix
df_x = df_final.loc[:, df_final.columns != 'target'] #drop the qualitative column
df_y = df_final.target
df_x.corr()

corr = df_x.corr() #high correlation exist, need use PCA to choose features


app_ui = ui.page_fluid(
    ui.h2("Please wait 3-5 minutes for visualisation"),
ui.h2("                                                                                          "),
          ui.h2("Dataset Overview"),
ui.h3("Dataset Name: Wisconsin Breast Cancer Diagnostic Dataset."),
ui.h3("Objective: To predict whether a tumor is benign or malignant."),
ui.h3("Number of Samples: 500    Number of features: 30"),
ui.h3("Feature Type: All features are numeric and describe the morphological characteristics of the tumor such as radius, texture, circumference, area, smoothness, symmetry, etc."),
ui.h3("Target Variable: dichotomous labels, 1 for benign and 0 for malignant."),
ui.h2("                                                                                          "),
          ui.h2("dataset Presentation (feathers and target)"),
          ui.output_table("data_head"),
ui.h2("schematic data distribution(selecting the first two features)"),
ui.h3("This is just a visualization, not all the features dimensions will be shown here"),
ui.output_plot("schematic", width="1800px", height="1800px"),
ui.h2("detection of autocorrelation between features"),
          ui.output_plot("corr_heatmap", width="1800px", height="1800px"),
ui.h3("Some negative effects of high correlation: Model overfitting, Decreased model stability, Waste of computational resources, etc."),
          ui.h3("Because of the strong correlation between some features, we need to screen the principal components using the PCA algorithm."),
          ui.h3("We need to split the test data before normalizing and pca."),
          ui.h3("Here you can define the proportion of the training set, the maximum number of features to be considered and complexity of the decision tree."),
          ui.h2("                                                                                          "),
          ui.input_slider("split_ratio", "train dataset ratio", min=0.1, max=0.9, value=0.8, width="800px"),
          ui.input_slider("maximum_split", "maximum number of features", min=2, max=7, value=4, width="800px"),
          ui.input_slider("complexity", "complexity of tree", min=1, max=7, value=3, width="800px"),
          ui.h2("                                                                                          "),
          ui.input_action_button("train", "train data"),
    ui.h2("                                                                                          "),
    ui.h2("Decision Tree model"),
    ui.h3("model result - confusion table"),
          ui.output_table("accuracy"),# 展示准确率
    ui.h3("model result - accuracy score"),
          ui.output_text("score"),
    ui.h3("Decision Tree Display"),
          ui.output_plot("tree_pic", width="1800px", height="1800px"),

    ui.h2("Random forest model"),
    ui.h3("model result - confusion table"),
          ui.output_table("accuracy_forest"),
    ui.h3("model result - accuracy score"),
          ui.output_text("score_forest"),
    ui.h3("Importance of each feature"),
          ui.output_plot("forest_pic", width="1800px", height="1800px"),

    ui.h2("principal component analysis (PCA) - You can know the characteristics corresponding to PC components"),
          ui.output_table("pca_analysis"),


    )



def server(input, output, session):
    @output
    @render.table
    def data_head():
        return df_final.head()

    @output
    @render.plot
    def schematic():
        mean_radius = df_x[["mean radius"]]
        mean_texture = df_x[["mean texture"]]
        plt.figure(figsize=[16, 33])
        plt.scatter(mean_radius[df_y == 0], mean_texture[df_y == 0], color='red', label='Malignant (0)', alpha=0.6)
        plt.scatter(mean_radius[df_y == 1], mean_texture[df_y == 1], color='blue', label='Benign (1)', alpha=0.6)
        plt.title("Breast Cancer Dataset: Mean Radius vs Mean Texture")
        plt.xlabel("Mean Radius")
        plt.ylabel("Mean Texture")
        plt.legend()
        return plt.gcf()

    @output
    @render.plot
    def corr_heatmap():
        plt.figure(figsize=[16, 33])
        plt.title("A Heatmap for Correlation Matrix")
        sns.heatmap(corr, annot=True, cmap="viridis")
        return plt.gcf()

    @output
    @render.table
    def accuracy():

        input.train()

        # 获取用户选择的分割比例
        split_ratio = input.split_ratio()
        maximum_split = input.maximum_split()
        complexity = input.complexity()

        prop = \
            int(len(df_final) * split_ratio)

        X_Train = df_x[: prop]  # First 80% of the data
        X_Test = df_x[prop:]  # Remaining 20% of the data

        Y_Train = df_y[:prop]
        Y_Test = df_y[prop:]

        Xtrains = X_Train - np.asarray(
            X_Train.mean(0))  # scale features before applying logistic regression with penalty
        Xtrain_scale = X_Train.std(0)
        Xtrains = Xtrains / np.asarray(Xtrain_scale)
        Xtests = (X_Test - np.asarray(X_Train.mean(0))) / np.asarray(
            Xtrain_scale)  # note that we use training mean and std to scale test set

        # PCA
        # This is not involve using test data's information to train the model
        pca = PCA(n_components=0.95)  # 95% variance
        Xtrains_pca = pca.fit_transform(Xtrains)
        Xtests_pca = pca.transform(Xtests)

        tuned_parameters = [{"max_features": list(range(1, maximum_split + 1)), "ccp_alpha": [10 ** (-i) for i in range(0, complexity)]}]
        treeCV = GridSearchCV(DecisionTreeClassifier(random_state=67), tuned_parameters, scoring='accuracy', cv=10)
        # more details see https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        treeCV.fit(Xtrains_pca, Y_Train)
        ypred_rf = treeCV.predict(Xtests_pca)
        yscores_tree = treeCV.predict_proba(Xtests_pca)  # obtain AUC value
        conf_matrix = confusion_table(ypred_rf, Y_Test)
        conf_matrix.insert(0, '', ["Predicted0", "Predicted1"])

        return conf_matrix

    @output
    @render.text
    def score():

        input.train()

        # 获取用户选择的分割比例
        split_ratio = input.split_ratio()
        maximum_split = input.maximum_split()
        complexity = input.complexity()

        prop = \
            int(len(df_final) * split_ratio)

        X_Train = df_x[: prop]  # First 80% of the data
        X_Test = df_x[prop:]  # Remaining 20% of the data

        Y_Train = df_y[:prop]
        Y_Test = df_y[prop:]

        Xtrains = X_Train - np.asarray(
            X_Train.mean(0))  # scale features before applying logistic regression with penalty
        Xtrain_scale = X_Train.std(0)
        Xtrains = Xtrains / np.asarray(Xtrain_scale)
        Xtests = (X_Test - np.asarray(X_Train.mean(0))) / np.asarray(
            Xtrain_scale)  # note that we use training mean and std to scale test set

        # PCA
        # This is not involve using test data's information to train the model
        pca = PCA(n_components=0.95)  # 95% variance
        Xtrains_pca = pca.fit_transform(Xtrains)
        Xtests_pca = pca.transform(Xtests)

        tuned_parameters = [{"max_features": list(range(1, maximum_split + 1)),
                             "ccp_alpha": [10 ** (-i) for i in range(0, complexity)]}]
        treeCV = GridSearchCV(DecisionTreeClassifier(random_state=67), tuned_parameters, scoring='accuracy', cv=10)
        # more details see https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        treeCV.fit(Xtrains_pca, Y_Train)
        ypred_rf = treeCV.predict(Xtests_pca)
        yscores_tree = treeCV.predict_proba(Xtests_pca)  # obtain AUC value
        conf_matrix = confusion_table(ypred_rf, Y_Test)
        conf_matrix.insert(0, '', ["Predicted0", "Predicted1"])
        yscores_tree = treeCV.predict_proba(Xtests_pca)  # obtain AUC value
        roc_auc_score(Y_Test, yscores_tree[:, 1])

        return roc_auc_score(Y_Test, yscores_tree[:, 1])

    @output
    @render.plot
    def tree_pic():

        input.train()

        # 获取用户选择的分割比例
        split_ratio = input.split_ratio()
        maximum_split = input.maximum_split()
        complexity = input.complexity()

        prop = \
            int(len(df_final) * split_ratio)

        X_Train = df_x[: prop]  # First 80% of the data
        X_Test = df_x[prop:]  # Remaining 20% of the data

        Y_Train = df_y[:prop]
        Y_Test = df_y[prop:]

        Xtrains = X_Train - np.asarray(
            X_Train.mean(0))  # scale features before applying logistic regression with penalty
        Xtrain_scale = X_Train.std(0)
        Xtrains = Xtrains / np.asarray(Xtrain_scale)
        Xtests = (X_Test - np.asarray(X_Train.mean(0))) / np.asarray(
            Xtrain_scale)  # note that we use training mean and std to scale test set

        # PCA
        # This is not involve using test data's information to train the model
        pca = PCA(n_components=0.95)  # 95% variance
        Xtrains_pca = pca.fit_transform(Xtrains)
        Xtests_pca = pca.transform(Xtests)

        # visualise tree
        dt = DecisionTreeClassifier(ccp_alpha=10 ** (-complexity), max_features=maximum_split, random_state=67)
        dt_vis = dt.fit(Xtrains_pca, Y_Train)
        fn = range(Xtrains_pca.shape[1])
        cn = ['1', '2']
        fig, axes = subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=300)
        tree.plot_tree(dt_vis,
                       feature_names=fn,
                       class_names=cn,
                       filled=True)

        return fig

    @output
    @render.table
    def accuracy_forest():
        input.train()

        # 获取用户选择的分割比例
        split_ratio = input.split_ratio()
        maximum_split = input.maximum_split()
        complexity = input.complexity()

        prop = \
            int(len(df_final) * split_ratio)

        X_Train = df_x[: prop]  # First 80% of the data
        X_Test = df_x[prop:]  # Remaining 20% of the data

        Y_Train = df_y[:prop]
        Y_Test = df_y[prop:]

        Xtrains = X_Train - np.asarray(
            X_Train.mean(0))  # scale features before applying logistic regression with penalty
        Xtrain_scale = X_Train.std(0)
        Xtrains = Xtrains / np.asarray(Xtrain_scale)
        Xtests = (X_Test - np.asarray(X_Train.mean(0))) / np.asarray(
            Xtrain_scale)  # note that we use training mean and std to scale test set

        # PCA
        # This is not involve using test data's information to train the model
        pca = PCA(n_components=0.95)  # 95% variance
        Xtrains_pca = pca.fit_transform(Xtrains)
        Xtests_pca = pca.transform(Xtests)

        tuned_parameters = [{"max_features": list(range(1, maximum_split + 1)),
                             "ccp_alpha": [10 ** (-i) for i in range(0, complexity)]}]
        rfCV = GridSearchCV(RandomForestClassifier(n_estimators=500, bootstrap=True, oob_score=True, random_state=234),
                            tuned_parameters, scoring='accuracy', cv=10)
        # more details see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        rfCV.fit(Xtrains_pca, Y_Train)
        ypred_rf = rfCV.predict(Xtests_pca)
        conf_matrix = confusion_table(ypred_rf, Y_Test)
        conf_matrix.insert(0, '', ["Predicted0", "Predicted1"])

        return conf_matrix

    @output
    @render.text
    def score_forest():
        input.train()

        # 获取用户选择的分割比例
        split_ratio = input.split_ratio()
        maximum_split = input.maximum_split()
        complexity = input.complexity()

        prop = \
            int(len(df_final) * split_ratio)

        X_Train = df_x[: prop]  # First 80% of the data
        X_Test = df_x[prop:]  # Remaining 20% of the data

        Y_Train = df_y[:prop]
        Y_Test = df_y[prop:]

        Xtrains = X_Train - np.asarray(
            X_Train.mean(0))  # scale features before applying logistic regression with penalty
        Xtrain_scale = X_Train.std(0)
        Xtrains = Xtrains / np.asarray(Xtrain_scale)
        Xtests = (X_Test - np.asarray(X_Train.mean(0))) / np.asarray(
            Xtrain_scale)  # note that we use training mean and std to scale test set

        # PCA
        # This is not involve using test data's information to train the model
        pca = PCA(n_components=0.95)  # 95% variance
        Xtrains_pca = pca.fit_transform(Xtrains)
        Xtests_pca = pca.transform(Xtests)

        tuned_parameters = [{"max_features": list(range(1, maximum_split + 1)),
                             "ccp_alpha": [10 ** (-i) for i in range(0, complexity)]}]
        rfCV = GridSearchCV(RandomForestClassifier(n_estimators=500, bootstrap=True, oob_score=True, random_state=234),
                            tuned_parameters, scoring='accuracy', cv=10)
        # more details see https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        rfCV.fit(Xtrains_pca, Y_Train)
        ypred_rf = rfCV.predict(Xtests_pca)
        conf_matrix = confusion_table(ypred_rf, Y_Test)
        conf_matrix.insert(0, '', ["Predicted0", "Predicted1"])

        return accuracy_score(Y_Test,ypred_rf)

    @output
    @render.plot
    def forest_pic():
        input.train()

        # 获取用户选择的分割比例
        split_ratio = input.split_ratio()
        maximum_split = input.maximum_split()
        complexity = input.complexity()

        prop = \
            int(len(df_final) * split_ratio)

        X_Train = df_x[: prop]  # First 80% of the data
        X_Test = df_x[prop:]  # Remaining 20% of the data

        Y_Train = df_y[:prop]
        Y_Test = df_y[prop:]

        Xtrains = X_Train - np.asarray(
            X_Train.mean(0))  # scale features before applying logistic regression with penalty
        Xtrain_scale = X_Train.std(0)
        Xtrains = Xtrains / np.asarray(Xtrain_scale)
        Xtests = (X_Test - np.asarray(X_Train.mean(0))) / np.asarray(
            Xtrain_scale)  # note that we use training mean and std to scale test set

        # PCA
        # This is not involve using test data's information to train the model
        pca = PCA(n_components=0.95)  # 95% variance
        Xtrains_pca = pca.fit_transform(Xtrains)
        Xtests_pca = pca.transform(Xtests)

        rf = RandomForestClassifier(n_estimators=500, max_features=maximum_split, bootstrap=True, oob_score=True,
                                    random_state=0).fit(Xtrains_pca, Y_Train)
        rf_importances = rf.feature_importances_
        Xtrains_pca_df = pd.DataFrame(Xtrains_pca, columns=[f'PC{i + 1}' for i in range(Xtrains_pca.shape[1])])
        feature_names = Xtrains_pca_df.columns
        # plot the most important features
        index = np.argsort(rf_importances)
        forest_importances = pd.Series(rf_importances[index[-10:]], index=feature_names[index[-10:]])

        fig, ax = plt.subplots()
        forest_importances.plot.bar()
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()

        return fig


    @output
    @render.table
    def pca_analysis():
        # 获取用户选择的分割比例
        split_ratio = input.split_ratio()
        maximum_split = input.maximum_split()
        complexity = input.complexity()

        prop = \
            int(len(df_final) * split_ratio)

        X_Train = df_x[: prop]  # First 80% of the data
        X_Test = df_x[prop:]  # Remaining 20% of the data

        Y_Train = df_y[:prop]
        Y_Test = df_y[prop:]

        Xtrains = X_Train - np.asarray(
        X_Train.mean(0))  # scale features before applying logistic regression with penalty
        Xtrain_scale = X_Train.std(0)
        Xtrains = Xtrains / np.asarray(Xtrain_scale)
        Xtests = (X_Test - np.asarray(X_Train.mean(0))) / np.asarray(
        Xtrain_scale)  # note that we use training mean and std to scale test set

        # PCA
        # This is not involve using test data's information to train the model
        pca = PCA(n_components=0.95)  # 95% variance
        Xtrains_pca = pca.fit_transform(Xtrains)
        Xtests_pca = pca.transform(Xtests)

        original_feature_names = Xtrains.columns
        loadings = pca.components_
        loadings_df = pd.DataFrame(loadings, columns=original_feature_names)
        loadings_df.index = [f'PC{i + 1}' for i in range(loadings.shape[0])]
        top_features = [loadings_df.loc[pc].idxmax() for pc in loadings_df.index]
        pca_column_names = [f'PC{i + 1} ({top_features[i]})' for i in range(len(top_features))]
        Xtrains_pca_t = pd.DataFrame(Xtrains_pca, columns=pca_column_names)

        return Xtrains_pca_t.head()


# 创建Shiny应用
app = App(app_ui, server)
app.run(port=8005)