import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    with open('./result.txt', 'r') as f:
        data = f.readlines()

    data_name = ['pos', 'neg', 'loc']

    plt.figure()

    plt.plot()
