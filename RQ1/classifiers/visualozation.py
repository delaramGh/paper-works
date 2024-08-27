if 1:
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load the data from the CSV file
    file_path = 'C:\\Users\\ASUS\\Desktop\\research\\mitacs project\\paper experiments\\RQ1\\results\\cifar dataset\\exp1_D2_configuration_0_5.csv'
    data = pd.read_csv(file_path)


    cleaned_data = data.dropna()


    plt.figure(figsize=(10, 6))

    unique_models = cleaned_data['model'].unique()
    color_list = ['red', 'green', 'blue', 'pink']#, 'orange']
    new_model_color_map = {model: color_list[i % len(color_list)] for i, model in enumerate(unique_models)}


    # Define a list of markers for the different models
    marker_list = ['o', 's', '^', 'p', 'D']  # Circle, Square, Triangle, Pentagon, Diamond
    model_marker_map = {model: marker_list[i % len(marker_list)] for i, model in enumerate(unique_models)}

    # Create the updated scatter plot with new markers
    plt.figure(figsize=(10, 6))
    for i, (model, group) in enumerate(cleaned_data.groupby('model')):
        plt.scatter(group['config'], group['accuracy'], color=new_model_color_map[model], marker=model_marker_map[model], s=100, label=model)

    plt.xlabel('Configuration')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy by Configuration')
    plt.legend(title='Models')
    plt.xticks(rotation=25, fontsize=7)  # Rotate labels if needed

    plt.show()

if 0:
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    # Load the data from the uploaded CSV file
    file_path = 'EXP1_resultv2.csv'
    data = pd.read_csv(file_path)

    # Create a violin plot
    plt.figure(figsize=(12, 6))
    sns.violinplot(data=data, x='model', y='accuracy', palette="Set3")
    plt.title('Violin Plot of Model Accuracies')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.show()


