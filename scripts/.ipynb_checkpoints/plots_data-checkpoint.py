{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "94e70968",
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "b7883907",
   "metadata": {},
   "outputs": [],
   "source": [
    "def plot_hist(df:pd.DataFrame, column: str, color:str)-> None:\n",
    "    sns.displot(data=df, x=column, color=color, kde=True, height=8, aspect=2)\n",
    "    plt.title(f'Distribution of {column}', size=20, fontweight='bold')\n",
    "    plt.show()  \n",
    "def plot_count(df:pd.DataFrame, column:str)->None:\n",
    "    plt.figure(figsize=(11,7))\n",
    "    sns.countplot(df, hue=column)\n",
    "    plt.title(f'Distribution of {column}', size=20, fontweight='bold')\n",
    "    plt.show()\n",
    "def plot_bar(df:pd.DataFrame, _col: str, y_col: str, title: str, xlabel: str, ylabel: str)->None:\n",
    "    plt.figure(figsize=(11, 7))\n",
    "    sns.barplot(data=df, x=x_col, y=y_col)\n",
    "    plt.title(title, size=20)\n",
    "    plt.xticks(rotation=75, fontsize=14)\n",
    "    plt.yticks(fontsize=14)\n",
    "    plt.xlabel(xlabel, fontsize=16)\n",
    "    plt.ylabel(ylabel, fontsize=16)\n",
    "def plot_heatmap(df: pd.DataFrame, title: str, cbar=False) -> None:\n",
    "    plt.figure(figsize=(11, 7))\n",
    "    sns.heatmap(df, annot=True, cmap='viridis', vmin=0, vmax=1, fmt='.2f', linewidths=.7, cbar=cbar)\n",
    "    plt.title(title, size=18, fontweight='bold')\n",
    "    plt.show()\n",
    "        \n",
    "def plot_box(df:pd.DataFrame, x_col:str, title:str) -> None:\n",
    "    plt.figure(figsize=(11, 7))\n",
    "    sns.boxplot(data=df, x=x_col)\n",
    "    plt.title(title, size=20)\n",
    "    plt.xticks(rotation=75, fontsize=14)\n",
    "    plt.show()\n",
    "        \n",
    "def plot_box_multi(df:pd.DataFrame, x_col:str, y_col:str, title:str)->None:\n",
    "    plt.figure(figsize=(11, 7))\n",
    "    sns.boxplot(data=df, x=x_col, y=y_col)\n",
    "    plt.title(title, size=20)\n",
    "    plt.xticks(rotation=75, fontsize=14)\n",
    "    plt.yticks(fontsize=14)\n",
    "    plt.show()    \n",
    "def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, title: str, hue: str, style: str) -> None:\n",
    "    plt.figure(figsize=(11, 7))\n",
    "    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, style=style)\n",
    "    plt.title(title, size=20)\n",
    "    plt.xticks(fontsize=14)\n",
    "    plt.yticks(fontsize=14)\n",
    "    plt.show()    \n",
    "def plot_pi(data, labels, title) -> None:\n",
    "    plt.figure(figsize=(11, 7))\n",
    "    colors = sns.color_palette('bright')\n",
    "    plt.pie(data, labels=labels, colors=colors, autopct='%.0f%%')\n",
    "    plt.title(title, size=20)\n",
    "    plt.show()    \n",
    "def figure_att(fig, title, titlex, titley, size, sizexy, weight)-> None:\n",
    "    fig.set_title(title, size=size, weight=weight)\n",
    "    figure.set_xlabel(titlex, size=sizexy, weight=weight)\n",
    "    figure.set_ylabel(titley, size=sizexy, weight=weight)           "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6e248fec",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
