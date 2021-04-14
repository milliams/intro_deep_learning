import seaborn as sns
iris = sns.load_dataset("iris")
f = sns.pairplot(data=iris, hue="species", markers=["o", "s", "D"])
f.savefig("Iris_dataset_scatterplot.png")
