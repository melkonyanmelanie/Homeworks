This project implements two popular malgorithms: **Epsilon-Greedy** and **Thompson Sampling**. 

**1. Install the Requirements**

To get started, you need to install all the necessary dependencies. A **requirements.txt** file is provided to help you install the required packages.

Simply run the following command in your terminal:

**pip install -r requirements.txt**


This will automatically install the necessary Python libraries and dependencies required for this project.

**2. Project Overview**

The main code is contained in the **bandit.py** file, which implements the Epsilon-Greedy and Thompson Sampling algorithms.

**Epsilon-Greedy** uses a decaying epsilon value to balance between exploration and exploitation, while **Thompson Sampling** utilizes Gaussian rewards with known precision to estimate the mean rewards of the arms.

**3. Project Structure**
- bandit.py               # Core implementation of the multi-armed bandit algorithms (Epsilon-Greedy & Thompson Sampling)
- requirements.txt        # Lists all required Python dependencies for the project.
- example_run/            # Contains an example of a run from the bandit.py script, showing output and performance metrics.
- suggestions.pdf         # A PDF file that provides additional suggestions for improving the experiment, including **batch simulations**, **stochastic rewards**, and an **adaptive "cool-off" strategy** for bandit exploration.

**4. How to Run the Code**

After installing the dependencies from requirements.txt, you can run the experiment by executing the bandit.py file.

**python bandit.py**


This will run the Epsilon-Greedy and Thompson Sampling algorithms on Bandit_Reward=[1, 2, 3, 4] for 20,000 trials. The results, including cumulative rewards and regrets, will be logged, and the data will be saved to a CSV file.

**5. Output**

Logs: The program logs cumulative rewards and regrets, along with other key metrics, to the console using loguru for easier tracking of experiment performance.

Plots: Learning curves and performance comparison plots will be saved as PNG files (learning_eg.png, learning_ts.png, comparison.png).

CSV: The results (bandit selections and rewards) will be saved in the rewards_log.csv file for future analysis.

6. Additional Suggestions

To further improve the analysis, the suggestions.pdf file includes recommendations on:
