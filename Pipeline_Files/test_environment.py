import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def test_environment():
    print("Testing NumPy...")
    arr = np.array([1, 2, 3, 4, 5])
    print("NumPy array created:", arr)
    
    print("\nTesting Pandas...")
    df = pd.DataFrame({
        'A': np.random.rand(5),
        'B': np.random.rand(5)
    })
    print("Pandas DataFrame created:\n", df)
    
    print("\nTesting Matplotlib and Seaborn...")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='A', y='B')
    plt.title("Test Plot")
    plt.savefig('test_plot.png')
    plt.close()
    print("Plot saved as 'test_plot.png'")
    
    print("\nAll basic tests passed! Your environment is working correctly.")

if __name__ == "__main__":
    test_environment() 