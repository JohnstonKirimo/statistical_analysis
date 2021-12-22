{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Sample script used to perform statistical analysis "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 1: Importing libraries and reading data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>obs</th>\n",
       "      <th>store_id</th>\n",
       "      <th>week</th>\n",
       "      <th>conversion_rate</th>\n",
       "      <th>adoption_rate</th>\n",
       "      <th>province</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1761</td>\n",
       "      <td>1</td>\n",
       "      <td>0.108245</td>\n",
       "      <td>0.212304</td>\n",
       "      <td>ON</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>2</td>\n",
       "      <td>1761</td>\n",
       "      <td>2</td>\n",
       "      <td>0.106241</td>\n",
       "      <td>0.326634</td>\n",
       "      <td>ON</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "      <td>1761</td>\n",
       "      <td>3</td>\n",
       "      <td>0.081902</td>\n",
       "      <td>0.056729</td>\n",
       "      <td>ON</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>3</td>\n",
       "      <td>4</td>\n",
       "      <td>1761</td>\n",
       "      <td>4</td>\n",
       "      <td>0.227862</td>\n",
       "      <td>0.065730</td>\n",
       "      <td>ON</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>4</td>\n",
       "      <td>5</td>\n",
       "      <td>1761</td>\n",
       "      <td>5</td>\n",
       "      <td>0.166521</td>\n",
       "      <td>0.031871</td>\n",
       "      <td>ON</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   obs  store_id  week  conversion_rate  adoption_rate province\n",
       "0    1      1761     1         0.108245       0.212304       ON\n",
       "1    2      1761     2         0.106241       0.326634       ON\n",
       "2    3      1761     3         0.081902       0.056729       ON\n",
       "3    4      1761     4         0.227862       0.065730       ON\n",
       "4    5      1761     5         0.166521       0.031871       ON"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test = pd.read_csv('/users/johnstonkirimo/projects/stats/data/test_dataset.csv')\n",
    "test.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Step 2: Getting a a high level overview of the dataset "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(300, 6)"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#rows and columns \n",
    "\n",
    "test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "obs                  int64\n",
       "store_id             int64\n",
       "week                 int64\n",
       "conversion_rate    float64\n",
       "adoption_rate      float64\n",
       "province            object\n",
       "dtype: object"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#columns and their data types\n",
    "\n",
    "test.dtypes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>conversion_rate</th>\n",
       "      <th>adoption_rate</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>count</td>\n",
       "      <td>300.000000</td>\n",
       "      <td>300.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>mean</td>\n",
       "      <td>0.218314</td>\n",
       "      <td>0.275981</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>std</td>\n",
       "      <td>0.086027</td>\n",
       "      <td>0.209942</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>min</td>\n",
       "      <td>-0.014476</td>\n",
       "      <td>0.003113</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>25%</td>\n",
       "      <td>0.155195</td>\n",
       "      <td>0.113982</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>50%</td>\n",
       "      <td>0.214930</td>\n",
       "      <td>0.227539</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>75%</td>\n",
       "      <td>0.282741</td>\n",
       "      <td>0.384571</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>max</td>\n",
       "      <td>0.418831</td>\n",
       "      <td>0.966212</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       conversion_rate  adoption_rate\n",
       "count       300.000000     300.000000\n",
       "mean          0.218314       0.275981\n",
       "std           0.086027       0.209942\n",
       "min          -0.014476       0.003113\n",
       "25%           0.155195       0.113982\n",
       "50%           0.214930       0.227539\n",
       "75%           0.282741       0.384571\n",
       "max           0.418831       0.966212"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#statistical summary of conversion rate and adoption rate\n",
    "\n",
    "test.describe()[['conversion_rate','adoption_rate']]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Keep only useful columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>store_id</th>\n",
       "      <th>week</th>\n",
       "      <th>conversion_rate</th>\n",
       "      <th>adoption_rate</th>\n",
       "      <th>province</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>0</td>\n",
       "      <td>1761</td>\n",
       "      <td>1</td>\n",
       "      <td>0.108245</td>\n",
       "      <td>0.212304</td>\n",
       "      <td>ON</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>1</td>\n",
       "      <td>1761</td>\n",
       "      <td>2</td>\n",
       "      <td>0.106241</td>\n",
       "      <td>0.326634</td>\n",
       "      <td>ON</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <td>2</td>\n",
       "      <td>1761</td>\n",
       "      <td>3</td>\n",
       "      <td>0.081902</td>\n",
       "      <td>0.056729</td>\n",
       "      <td>ON</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   store_id  week  conversion_rate  adoption_rate province\n",
       "0      1761     1         0.108245       0.212304       ON\n",
       "1      1761     2         0.106241       0.326634       ON\n",
       "2      1761     3         0.081902       0.056729       ON"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test = test[['store_id','week','conversion_rate','adoption_rate','province']]\n",
    "test.head(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Does each store have five-week data?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "store_id\n",
       "45      5\n",
       "135     5\n",
       "251     5\n",
       "388     5\n",
       "402     5\n",
       "594     5\n",
       "1036    5\n",
       "1516    5\n",
       "1578    5\n",
       "1761    5\n",
       "1860    5\n",
       "2238    5\n",
       "2371    5\n",
       "2657    5\n",
       "2763    5\n",
       "2768    5\n",
       "2960    5\n",
       "3003    5\n",
       "3161    5\n",
       "3195    5\n",
       "3631    5\n",
       "3762    5\n",
       "4078    5\n",
       "4127    5\n",
       "4290    5\n",
       "4492    5\n",
       "4697    5\n",
       "4819    5\n",
       "5097    5\n",
       "5217    5\n",
       "5499    5\n",
       "5501    5\n",
       "6205    5\n",
       "6272    5\n",
       "6448    5\n",
       "6571    5\n",
       "6650    5\n",
       "6950    5\n",
       "6999    5\n",
       "7040    5\n",
       "7104    5\n",
       "7630    5\n",
       "7638    5\n",
       "7722    5\n",
       "7942    5\n",
       "7991    5\n",
       "8159    5\n",
       "8161    5\n",
       "8585    5\n",
       "8592    5\n",
       "8823    5\n",
       "8872    5\n",
       "9044    5\n",
       "9166    5\n",
       "9543    5\n",
       "9832    5\n",
       "9839    5\n",
       "9932    5\n",
       "9958    5\n",
       "9975    5\n",
       "Name: week, dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test.groupby('store_id').week.size()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Check how many stores have negative conversion rate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>store_id</th>\n",
       "      <th>week</th>\n",
       "      <th>conversion_rate</th>\n",
       "      <th>adoption_rate</th>\n",
       "      <th>province</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <td>88</td>\n",
       "      <td>3195</td>\n",
       "      <td>4</td>\n",
       "      <td>-0.014476</td>\n",
       "      <td>0.104227</td>\n",
       "      <td>ON</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    store_id  week  conversion_rate  adoption_rate province\n",
       "88      3195     4        -0.014476       0.104227       ON"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "test.query('conversion_rate < 0')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Plotting the adoption rate to see how it's distributed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7fecb5ea2d90>"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "test.adoption_rate.hist()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Plotting the conversion rate to see how it's distributed"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7fecb64246d0>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "test.conversion_rate.hist()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Scatter plot to check for any correlations between adoption and conversion rates"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.axes._subplots.AxesSubplot at 0x7fecb6554750>"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "test.plot(kind='scatter', x='adoption_rate', y='conversion_rate')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "#### Hypothesis testing:\n",
    "\n",
    "- Using Z-test because conversion rate is normally distributed but adoption isn't and the data set size is large enough\n",
    "\n",
    "    - High adoption rate >= 0.5\n",
    "    - Low adoption rate < 0.5  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "#High adoption locations\n",
    "\n",
    "higher_adoption = test.query('adoption_rate >= 0.5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "50"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(higher_adoption)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "#low adoption locations\n",
    "\n",
    "lower_adoption = test.query('adoption_rate < 0.5')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "250"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(lower_adoption)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "####  Significance level: 0.05  (an indication of how far the adoption rate has to deviate to reject the null hypothesis)\n",
    "- P-value: To compare the p-value to significance level: If it's less than significance level, then reject hypothesis, otherwise accept it\n",
    "- Null_hypothesis (H0): Mean_Ha > Mean_La\n",
    "- Alternative_hypothesis(H1): Mean_Ha <= Mean_La "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.2734334229\n"
     ]
    }
   ],
   "source": [
    "mean_Ha = higher_adoption.conversion_rate.mean()\n",
    "print(mean_Ha)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.2072896589759999\n"
     ]
    }
   ],
   "source": [
    "mean_La = lower_adoption.conversion_rate.mean()\n",
    "print(mean_La)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Importing scipy and stats models"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "from scipy import stats\n",
    "from statsmodels.stats import weightstats as stests"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Calculating p-value"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.4999999999999974\n",
      "accept null hypothesis\n"
     ]
    }
   ],
   "source": [
    "ztest,pval = stests.ztest(x1=higher_adoption.conversion_rate, x2=lower_adoption.conversion_rate, \n",
    "                           value=mean_Ha - mean_La, alternative='smaller')\n",
    "print(float(pval))\n",
    "if pval<0.05:\n",
    "    print(\"reject null hypothesis\")\n",
    "else:\n",
    "    print(\"accept null hypothesis\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Key Insight:\n",
    "  - We accept the null hypothesis (Mean_Ha > Mean_La)\n",
    "  - This means stores with higher adoption rate also have a higher conversion rate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
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
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
